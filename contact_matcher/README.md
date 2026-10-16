# Contact matcher

Adds the emails and phone numbers from a contacts database to a distributor list.

```
pip install -r requirements.txt
python match_contacts.py contacts.xlsx distributors.xlsx
```

This writes `distributors_with_contacts.xlsx` next to the distributor file. Use
`-o other_name.xlsx` to pick the name. `.csv` inputs and outputs also work.

## Input columns

| Contacts database | Distributor list |
|---|---|
| Contact Name | Distributor Name |
| Address | Address |
| Address City | City |
| Address State | ST |
| Address Zipcode | Zip |

Header names are matched without regard to case or extra spaces. If a header
is named differently, edit `CONTACT_KEY_COLUMNS` / `DISTRIBUTOR_KEY_COLUMNS`
at the top of the script.

## Added columns

`Match Type`, `Match Count`, `Matched Contact ID`, `Matched Contact Name`,
`Matched Address`, `Contact Email`, `Address Email`, `Cp Name`, `Cp Email`,
`Address Phone`, `Address Alt Phone 1`, `Address Alt Phone 2`, `Cp Phone`,
`Cp Phone 2`. Edit `COPY_COLUMNS` in the script to change this list.

When one distributor matches several contacts (for example, several people at
the same company), their distinct values are joined with `; `.

## How rows are matched

Names, addresses, cities, states and ZIPs are normalized before comparing.
Case, punctuation, Inc/LLC/Co suffixes, Street/St, Suite/Ste/#, full state names,
ZIP+4 and ZIPs that lost their leading zero all compare equal. Each
distributor gets the first rule that finds a match:

1. `name+address+zip`
2. `name+zip`
3. `name+city+state`
4. `fuzzy name+zip`: a name at least 90% similar in the same ZIP
   (`--fuzzy-threshold 95` makes it stricter, `0` turns it off)
5. `address+zip`: same address, different name. This can be a renamed company
   or a different business in the same building, so check these rows.

Rows with no match get `no match`. Filter the `Match Type` column in Excel to
review the looser rules.

## Speed

On a test set of 100,000 contacts and 10,000 distributors the script takes
about 25 seconds with just pandas and openpyxl, and about 10 seconds with the
optional `python-calamine` and `rapidfuzz` packages installed. Most of that
time is spent reading the large Excel file.
