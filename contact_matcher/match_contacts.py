#!/usr/bin/env python3
"""Copy contact emails and phone numbers from a contacts database onto a distributor list.

Usage:
    python match_contacts.py CONTACTS.xlsx DISTRIBUTORS.xlsx [-o OUTPUT.xlsx]

Each distributor row is matched against the contacts database with the first
rule below that finds a hit. Every comparison uses normalized text, so case,
punctuation, "Inc"/"LLC" suffixes, "Street"/"St", "Suite"/"#", full state names,
ZIP+4 codes and ZIPs that lost their leading zero in Excel don't block a match.

    1. name + address + zip
    2. name + zip
    3. name + city + state
    4. fuzzy name + zip     (similar name in the same zip, see --fuzzy-threshold)
    5. address + zip        (same address, different name; worth a review)

When several contacts match one distributor, their distinct values are joined
with "; ". The output keeps every distributor column and adds the contact
columns plus "Match Type" and "Match Count" so matches can be filtered and
reviewed in Excel.
"""

import argparse
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd

# Columns in the contacts database (header lookup ignores case and extra spaces).
CONTACT_KEY_COLUMNS = {
    "name": "Contact Name",
    "address": "Address",
    "city": "Address City",
    "state": "Address State",
    "zip": "Address Zipcode",
}

# Columns in the distributor list.
DISTRIBUTOR_KEY_COLUMNS = {
    "name": "Distributor Name",
    "address": "Address",
    "city": "City",
    "state": "ST",
    "zip": "Zip",
}

# Contact columns copied onto each matched distributor row: {source: output name}.
# Remove or add entries here to change what ends up in the output.
COPY_COLUMNS = {
    "Contact ID": "Matched Contact ID",
    "Contact Name": "Matched Contact Name",
    "Address": "Matched Address",
    "Contact Email": "Contact Email",
    "Address Email": "Address Email",
    "Cp Name": "Cp Name",
    "Cp Email": "Cp Email",
    "Address Phone": "Address Phone",
    "Address Alt Phone 1": "Address Alt Phone 1",
    "Address Alt Phone 2": "Address Alt Phone 2",
    "Cp Phone": "Cp Phone",
    "Cp Phone 2": "Cp Phone 2",
}

FUZZY_RULE = "fuzzy name+zip"
MATCH_RULES = [
    ("name+address+zip", ("name", "address", "zip")),
    ("name+zip", ("name", "zip")),
    ("name+city+state", ("name", "city", "state")),
    (FUZZY_RULE, ("name", "zip")),
    ("address+zip", ("address", "zip")),
]
NO_MATCH = "no match"

EXCEL_CELL_LIMIT = 32767

US_STATES = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR", "california": "CA",
    "colorado": "CO", "connecticut": "CT", "delaware": "DE", "district of columbia": "DC",
    "florida": "FL", "georgia": "GA", "hawaii": "HI", "idaho": "ID", "illinois": "IL",
    "indiana": "IN", "iowa": "IA", "kansas": "KS", "kentucky": "KY", "louisiana": "LA",
    "maine": "ME", "maryland": "MD", "massachusetts": "MA", "michigan": "MI",
    "minnesota": "MN", "mississippi": "MS", "missouri": "MO", "montana": "MT",
    "nebraska": "NE", "nevada": "NV", "new hampshire": "NH", "new jersey": "NJ",
    "new mexico": "NM", "new york": "NY", "north carolina": "NC", "north dakota": "ND",
    "ohio": "OH", "oklahoma": "OK", "oregon": "OR", "pennsylvania": "PA",
    "puerto rico": "PR", "rhode island": "RI", "south carolina": "SC",
    "south dakota": "SD", "tennessee": "TN", "texas": "TX", "utah": "UT", "vermont": "VT",
    "virginia": "VA", "washington": "WA", "west virginia": "WV", "wisconsin": "WI",
    "wyoming": "WY",
}

# Words dropped from company names before comparing them.
NAME_NOISE = {
    "the", "inc", "incorporated", "llc", "ltd", "limited", "co", "corp", "corporation",
    "company", "lp", "llp", "pllc", "pc",
}

ADDRESS_WORDS = {
    "street": "st", "avenue": "ave", "av": "ave", "road": "rd", "drive": "dr",
    "boulevard": "blvd", "lane": "ln", "court": "ct", "place": "pl", "parkway": "pkwy",
    "highway": "hwy", "freeway": "fwy", "expressway": "expy", "circle": "cir",
    "terrace": "ter", "trail": "trl", "square": "sq", "plaza": "plz", "center": "ctr",
    "route": "rte", "floor": "fl", "building": "bldg",
    "north": "n", "south": "s", "east": "e", "west": "w",
    "northeast": "ne", "northwest": "nw", "southeast": "se", "southwest": "sw",
    "suite": "unit", "ste": "unit", "apt": "unit", "apartment": "unit",
    "room": "unit", "rm": "unit",
}

CITY_WORDS = {"saint": "st", "fort": "ft", "mount": "mt"}

_NON_ALNUM = re.compile(r"[^a-z0-9]+")
_PO_BOX = re.compile(r"\b(p\s*o|post\s+office)\s+box\b")
_FLOAT_INT = re.compile(r"^(-?\d+)\.0+$")


def _tokens(text):
    return _NON_ALNUM.sub(" ", text.lower().replace(".", "").replace("'", "")).split()


def norm_name(text):
    tokens = _tokens(text.replace("&", " and "))
    kept = [t for t in tokens if t not in NAME_NOISE]
    return "".join(kept or tokens)


def norm_address(text):
    text = text.lower().replace(".", "").replace("#", " unit ")
    text = _PO_BOX.sub(" pobox ", text)
    words = [ADDRESS_WORDS.get(t, t) for t in _tokens(text)]
    # "Suite #100" -> "unit unit 100" -> "unit 100"
    words = [w for i, w in enumerate(words) if not (w == "unit" and i and words[i - 1] == "unit")]
    return " ".join(words)


def norm_city(text):
    return "".join(CITY_WORDS.get(t, t) for t in _tokens(text))


def norm_state(text):
    state = " ".join(_tokens(text))
    return US_STATES.get(state, state.upper())


def norm_zip(text):
    digits = re.sub(r"\D", "", text)
    if digits and re.search(r"[A-Za-z]", text):  # non-US postal code, e.g. Canadian "M5V 3L9"
        return re.sub(r"[^A-Z0-9]", "", text.upper())
    if len(digits) < 3:
        return ""
    # Restore leading zeros Excel dropped and cut ZIP+4 down to the 5-digit ZIP.
    return digits.zfill(5) if len(digits) <= 5 else digits.zfill(9)[:5]


NORMALIZERS = {
    "name": norm_name,
    "address": norm_address,
    "city": norm_city,
    "state": norm_state,
    "zip": norm_zip,
}

try:
    from rapidfuzz import fuzz, process

    def best_fuzzy_match(query, choices, threshold):
        hit = process.extractOne(query, choices, scorer=fuzz.ratio, score_cutoff=threshold)
        return hit[0] if hit else None

except ImportError:
    from difflib import SequenceMatcher

    def best_fuzzy_match(query, choices, threshold):
        best, best_score = None, threshold / 100
        matcher = SequenceMatcher(autojunk=False)
        matcher.set_seq2(query)
        for choice in choices:
            matcher.set_seq1(choice)
            if matcher.real_quick_ratio() < best_score or matcher.quick_ratio() < best_score:
                continue
            score = matcher.ratio()
            if score >= best_score:
                best, best_score = choice, score
        return best


def read_table(path, sheet):
    if path.suffix.lower() in {".csv", ".txt"}:
        df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding_errors="replace")
    else:
        if sheet is None:
            sheet = 0
        elif sheet.isdigit():
            sheet = int(sheet)
        try:
            import python_calamine  # noqa: F401  optional, reads .xlsx several times faster
            engine = "calamine"
        except ImportError:
            engine = None
        df = pd.read_excel(path, sheet_name=sheet, engine=engine)
    df.columns = [" ".join(str(c).split()) for c in df.columns]
    return df.reset_index(drop=True)


def as_text(df):
    """Every cell as a trimmed string, with numbers like 2134.0 turned back into 2134."""
    text = df.astype(object).where(df.notna(), "").astype(str)
    return text.apply(lambda col: col.str.strip().str.replace(_FLOAT_INT, r"\1", regex=True))


def resolve_columns(df, wanted, label, required):
    lookup = {c.lower(): c for c in df.columns}
    found = {w: lookup[w.lower()] for w in wanted if w.lower() in lookup}
    missing = [w for w in wanted if w not in found]
    if missing:
        message = f"{label} has no column named: {', '.join(missing)}"
        if required:
            sys.exit(f"{message}\nColumns found: {', '.join(df.columns)}")
        print(f"Warning: {message} (skipped)")
    return found


def match_keys(text, key_columns, label):
    columns = resolve_columns(text, list(key_columns.values()), label, required=True)
    keys = {}
    for key, column in key_columns.items():
        series = text[columns[column]]
        uniques = series.unique()
        keys[key] = series.map(dict(zip(uniques, map(NORMALIZERS[key], uniques)))).tolist()
    return keys


def build_index(keys, fields):
    index = defaultdict(list)
    for position, key in enumerate(zip(*(keys[f] for f in fields))):
        if all(key):
            index[key].append(position)
    return index


def match_rows(contact_keys, distributor_keys, fuzzy_threshold):
    """Return (match type, contact row positions) for every distributor row."""
    results = [(NO_MATCH, [])] * len(distributor_keys["name"])
    for rule, fields in MATCH_RULES:
        if rule == FUZZY_RULE and fuzzy_threshold <= 0:
            continue
        index = build_index(contact_keys, fields)
        if rule == FUZZY_RULE:
            names_by_zip = defaultdict(list)
            for name, zip_code in index:
                names_by_zip[zip_code].append(name)
        for row, key in enumerate(zip(*(distributor_keys[f] for f in fields))):
            if results[row][0] != NO_MATCH or not all(key):
                continue
            if rule == FUZZY_RULE:
                name, zip_code = key
                best = best_fuzzy_match(name, names_by_zip.get(zip_code, []), fuzzy_threshold)
                key = (best, zip_code) if best else None
            if key in index:
                results[row] = (rule, index[key])
    return results


def combine(values):
    return "; ".join(dict.fromkeys(v for v in values if v))[:EXCEL_CELL_LIMIT]


def write_table(df, path):
    if path.suffix.lower() == ".csv":
        df.to_csv(path, index=False, encoding="utf-8-sig")
        return
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Distributors")
        sheet = writer.sheets["Distributors"]
        sheet.freeze_panes = "A2"
        sheet.auto_filter.ref = sheet.dimensions


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Add contact emails and phone numbers from a contacts database to a distributor list.")
    parser.add_argument("contacts", type=Path, help="contacts database (.xlsx, .xls or .csv)")
    parser.add_argument("distributors", type=Path, help="distributor list (.xlsx, .xls or .csv)")
    parser.add_argument("-o", "--output", type=Path,
                        help="output file (default: <distributors>_with_contacts.xlsx next to the input)")
    parser.add_argument("--contacts-sheet", help="sheet name or 0-based index (default: first sheet)")
    parser.add_argument("--distributors-sheet", help="sheet name or 0-based index (default: first sheet)")
    parser.add_argument("--fuzzy-threshold", type=float, default=90,
                        help="minimum name similarity (0-100) for the fuzzy rule; 0 turns it off (default: 90)")
    args = parser.parse_args(argv)
    for path in (args.contacts, args.distributors):
        if not path.is_file():
            sys.exit(f"File not found: {path}")

    output = args.output
    if output is None:
        suffix = ".csv" if args.distributors.suffix.lower() == ".csv" else ".xlsx"
        output = args.distributors.with_name(f"{args.distributors.stem}_with_contacts{suffix}")

    start = time.perf_counter()
    print(f"Reading {args.contacts} ...")
    contacts = as_text(read_table(args.contacts, args.contacts_sheet))
    print(f"Reading {args.distributors} ...")
    distributors = read_table(args.distributors, args.distributors_sheet)
    print(f"  {len(contacts):,} contacts, {len(distributors):,} distributors "
          f"({time.perf_counter() - start:.1f}s)")

    contact_keys = match_keys(contacts, CONTACT_KEY_COLUMNS, "Contacts file")
    distributor_keys = match_keys(as_text(distributors), DISTRIBUTOR_KEY_COLUMNS, "Distributors file")
    results = match_rows(contact_keys, distributor_keys, args.fuzzy_threshold)

    copy_columns = resolve_columns(contacts, list(COPY_COLUMNS), "Contacts file", required=False)
    output_df = distributors.copy()
    output_df["Match Type"] = [rule for rule, _ in results]
    output_df["Match Count"] = [len(rows) for _, rows in results]
    for source, column in copy_columns.items():
        values = contacts[column].tolist()
        output_df[COPY_COLUMNS[source]] = [combine(values[r] for r in rows) for _, rows in results]

    print(f"Writing {output} ...")
    try:
        write_table(output_df, output)
    except PermissionError:
        sys.exit(f"Could not write {output}. Is it open in Excel? Close it and run again.")

    counts = output_df["Match Type"].value_counts()
    print(f"\nDone in {time.perf_counter() - start:.1f}s")
    for rule in [r for r, _ in MATCH_RULES] + [NO_MATCH]:
        if rule in counts:
            print(f"  {rule:<18} {counts[rule]:>8,}")


if __name__ == "__main__":
    main()
