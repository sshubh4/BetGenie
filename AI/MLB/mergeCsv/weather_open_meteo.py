import csv

csv_file = "./teamgamelogs/WSN_boxscores.csv"  # change to your CSV path
row_index_to_check = 760    # change to the row index you want (0-based: 0 = header, 1 = first row)

with open(csv_file, newline='', encoding='utf-8') as f:
    reader = list(csv.reader(f))
    
    header = reader[0]
    target_row = reader[row_index_to_check]
    
    print(f"Header column count: {len(header)}")
    print(f"Row {row_index_to_check} column count: {len(target_row)}")
    print(f"Column count difference: {len(target_row) - len(header)}")
    print("Row data preview:")
    print(target_row)
