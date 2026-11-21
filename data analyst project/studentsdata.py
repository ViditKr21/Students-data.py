"""Student Management & Analytics System  (single file, self-repairing)

Covers: dataset (60 records), Python data structures, user-defined functions,
NumPy, Pandas, file handling, Add / Display / Search / Update / Delete,
summary report, data analysis insights, clear terminal output.
Run:  python student_management.py        (needs: pip install pandas numpy)
It creates / repairs students.csv by itself, so no file errors.
"""
import os
import random
from io import StringIO

import numpy as np
import pandas as pd

FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "students.csv")
SUBJECTS = ["Math", "Science", "English"]
NUMERIC = SUBJECTS + ["Attendance"]
BASE_COLS = ["Student_ID", "Name", "Department", "Gender", "Age", "City"] + NUMERIC
DEPTS = ["CSE", "ECE", "MECH", "CIVIL", "IT"]
CITIES = ["Hyderabad", "Chennai", "Bengaluru", "Mumbai", "Delhi", "Pune"]
FIRST = ["Peehu", "Diya", "Rohan", "Sneha", "Karthik", "Peehu", "Arjun", "Meera",
         "Vikram", "Ananya", "Rahul", "Kavya", "Suresh", "Divya", "Nikhil",
         "Pooja", "Aditya", "Isha", "Manoj", "Lakshmi"]
LAST = ["Sharma", "Reddy", "Kumar", "Nair", "Singh", "Choudhary"]


# ----------------------------- small helpers ------------------------------
def line(char="=", n=78):
    print(char * n)


def title(text):
    print()
    line()
    print(f"  {text}")
    line()


def grade(avg):
    for cutoff, g in [(90, "A+"), (80, "A"), (70, "B"), (60, "C"), (50, "D")]:
        if avg >= cutoff:
            return g
    return "F"


def ask_number(prompt, low=0, high=100):
    """Keep asking until the user enters a number between low and high."""
    while True:
        try:
            val = float(input(prompt))
            if low <= val <= high:
                return val
        except ValueError:
            pass
        print(f"  Please enter a number between {low} and {high}.")


# ------------------------------ dataset / file ----------------------------
def create_sample_dataset(n=60):
    """Create 60 sample records (used only if students.csv is missing/empty)."""
    random.seed(42)
    rows = []
    for i in range(1, n + 1):
        rows.append({
            "Student_ID": f"S{i:03d}",
            "Name": f"{FIRST[(i - 1) % len(FIRST)]} {random.choice(LAST)}",
            "Department": random.choice(DEPTS),
            "Gender": "F" if i % 2 == 0 else "M",
            "Age": random.randint(18, 23),
            "City": random.choice(CITIES),
            "Math": random.randint(35, 99),
            "Science": random.randint(35, 99),
            "English": random.randint(40, 99),
            "Attendance": random.randint(60, 100),
        })
    pd.DataFrame(rows).to_csv(FILE, index=False)
    print(f"[info] students.csv was missing - created a new one with {n} records.")


def read_file():
    """Read the CSV even with junk lines on top, quotes, or ; / tab separators."""
    if not os.path.exists(FILE) or os.path.getsize(FILE) == 0:
        create_sample_dataset()
    with open(FILE, encoding="utf-8-sig", errors="replace") as f:
        lines = f.read().splitlines()
    start = next((i for i, l in enumerate(lines) if "student_id" in l.lower()), None)
    if start is None:                       # unreadable file -> rebuild it
        print("[info] No 'Student_ID' header found - creating a fresh students.csv.")
        create_sample_dataset()
        return read_file()
    header = lines[start]
    sep = max([",", ";", "\t", "|"], key=header.count)
    df = pd.read_csv(StringIO("\n".join(lines[start:])), sep=sep,
                     skipinitialspace=True, on_bad_lines="skip")
    df.columns = df.columns.str.strip().str.strip("\"'").str.strip()
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    return df.dropna(how="all")


def repair_columns(df):
    """Add any missing column with sample values so the program never crashes."""
    random.seed(1)
    gens = {"Name": lambda: "Student", "Department": lambda: random.choice(DEPTS),
            "Gender": lambda: random.choice(["M", "F"]),
            "Age": lambda: random.randint(18, 23), "City": lambda: random.choice(CITIES),
            "Math": lambda: random.randint(35, 99), "Science": lambda: random.randint(35, 99),
            "English": lambda: random.randint(40, 99),
            "Attendance": lambda: random.randint(60, 100)}
    added = []
    if "Student_ID" not in df.columns:
        df["Student_ID"] = [f"S{i + 1:03d}" for i in range(len(df))]
        added.append("Student_ID")
    for col, gen in gens.items():
        if col not in df.columns:
            df[col] = [gen() for _ in range(len(df))]
            added.append(col)
    if added:
        print(f"[info] Missing column(s) {added} were added with SAMPLE values.")
    return df


def calculate(df):
    """Add Average and Grade columns."""
    df["Average"] = np.round(df[SUBJECTS].mean(axis=1), 2)
    df["Grade"] = df["Average"].apply(grade)
    return df


def load():
    df = repair_columns(read_file())
    df["Student_ID"] = df["Student_ID"].astype(str).str.strip()
    for c in NUMERIC + ["Age"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in NUMERIC:                                   # handle missing values
        df[c] = df[c].fillna(round(df[c].mean(), 1))
    df["Age"] = df["Age"].fillna(round(df["Age"].mean())).astype(int)
    for c in ["Name", "Department", "City", "Gender"]:
        df[c] = df[c].fillna("Unknown").astype(str).str.strip()
    save(df.copy())                                     # keep file clean
    return calculate(df)


def save(df):
    df.drop(columns=["Average", "Grade"], errors="ignore").to_csv(FILE, index=False)


def show(df):
    cols = ["Student_ID", "Name", "Department", "Age", "City",
            "Math", "Science", "English", "Attendance", "Average", "Grade"]
    print(df[cols].to_string(index=False))


def next_id(df):
    nums = pd.to_numeric(df["Student_ID"].str.extract(r"(\d+)")[0], errors="coerce")
    return f"S{int(nums.max() if nums.notna().any() else 0) + 1:03d}"


# --------------------------------- CRUD -----------------------------------
def add_record(df):
    title("ADD NEW RECORD")
    new_id = next_id(df)
    print(f"New Student ID: {new_id}")
    rec = {"Student_ID": new_id,
           "Name": input("Name: ").strip().title(),
           "Department": input("Department (CSE/ECE/MECH/CIVIL/IT): ").strip().upper(),
           "Gender": input("Gender (M/F): ").strip().upper(),
           "Age": int(ask_number("Age: ", 5, 100)),
           "City": input("City: ").strip().title()}
    for s in NUMERIC:
        rec[s] = ask_number(f"{s} (0-100): ")
    df = pd.concat([df.drop(columns=["Average", "Grade"]), pd.DataFrame([rec])],
                   ignore_index=True)
    save(df)
    print(f"\n✔ Record {new_id} added.")


def search_record(df):
    title("SEARCH RECORD")
    key = input("Enter Student ID or name: ").strip()
    res = df[(df["Student_ID"].str.upper() == key.upper()) |
             df["Name"].str.contains(key, case=False, na=False, regex=False)]
    if res.empty:
        print("\nNo record found.")
    else:
        print(f"\n{len(res)} record(s) found:")
        show(res)


def update_record(df):
    title("UPDATE RECORD")
    sid = input("Student ID: ").strip().upper()
    idx = df.index[df["Student_ID"].str.upper() == sid]
    if len(idx) == 0:
        print("Student ID not found.")
        return
    fields = ["Name", "Department", "Age", "City"] + NUMERIC
    field = input(f"Field ({'/'.join(fields)}): ").strip().title()
    if field not in fields:
        print("Invalid field.")
        return
    if field in NUMERIC + ["Age"]:
        value = ask_number("New value: ")
    else:
        value = input("New value: ").strip()
    df.loc[idx, field] = value
    save(df)
    print(f"\n✔ {field} updated for {sid}.")
    show(calculate(df).loc[idx])


def delete_record(df):
    title("DELETE RECORD")
    sid = input("Student ID: ").strip().upper()
    if sid not in df["Student_ID"].str.upper().values:
        print("Student ID not found.")
        return
    if input(f"Delete {sid}? (y/n): ").strip().lower() != "y":
        print("Cancelled.")
        return
    df = df[df["Student_ID"].str.upper() != sid].reset_index(drop=True)
    save(df)
    print(f"\n✔ Record {sid} deleted.")


# ----------------------- report / analysis / demos ------------------------
def summary_report(df):
    title("SUMMARY REPORT")
    print(f"Total students      : {len(df)}")
    print(f"Departments         : {df['Department'].nunique()}")
    print(f"Average age         : {df['Age'].mean():.1f}")
    print(f"Overall avg marks   : {df['Average'].mean():.2f}")
    print(f"Overall attendance  : {df['Attendance'].mean():.2f}%")
    print(f"Highest / Lowest avg: {df['Average'].max()} / {df['Average'].min()}")
    print("\nGrade distribution:")
    print(df["Grade"].value_counts().sort_index().to_string())
    print("\nDepartment-wise summary:")
    print(df.groupby("Department").agg(Students=("Student_ID", "count"),
                                       Avg_Marks=("Average", "mean"),
                                       Avg_Attendance=("Attendance", "mean")).round(2).to_string())


def data_analysis(df):
    title("DATA ANALYSIS - INSIGHTS (NumPy + Pandas)")
    marks = df[SUBJECTS].to_numpy()                          # NumPy 2D array
    print("NumPy statistics per subject:")
    for i, s in enumerate(SUBJECTS):
        col = marks[:, i]
        print(f"  {s:<8} sum={col.sum():7.0f}  mean={col.mean():6.2f}  max={col.max():5.1f}  "
              f"min={col.min():5.1f}  std={col.std():6.2f}")
    best, worst = df.loc[df["Average"].idxmax()], df.loc[df["Average"].idxmin()]
    dept = df.groupby("Department")["Average"].mean()
    corr = np.corrcoef(df["Attendance"], df["Average"])[0, 1] if len(df) > 1 else 0.0
    print("\nKey insights:")
    print(f"  1. Topper             : {best['Name']} ({best['Department']}) - {best['Average']}%")
    print(f"  2. Lowest performer   : {worst['Name']} ({worst['Department']}) - {worst['Average']}%")
    print(f"  3. Best department    : {dept.idxmax()} ({dept.max():.2f}%)")
    print(f"  4. Weakest department : {dept.idxmin()} ({dept.min():.2f}%)")
    print(f"  5. Strongest subject  : {df[SUBJECTS].mean().idxmax()}")
    print(f"  6. Weakest subject    : {df[SUBJECTS].mean().idxmin()}")
    print(f"  7. Attendance-marks correlation: {corr:.3f}")
    print(f"  8. Students below 75% attendance: {(df['Attendance'] < 75).sum()}")
    print(f"  9. Pass rate (avg >= 50): {(df['Average'] >= 50).mean() * 100:.1f}%")
    print("\nTop 5 students:")
    print(df.sort_values("Average", ascending=False).head(5)
          [["Student_ID", "Name", "Department", "Average", "Grade"]].to_string(index=False))


def concepts_demo(df):
    """Python data structures + NumPy + Pandas demonstrations."""
    title("PYTHON DATA STRUCTURES")
    names = df["Name"].head(5).tolist()                       # list
    names.append("Extra"); names.sort(); names.remove("Extra")
    print("List + methods      :", names)
    print("Tuple (min, max)    :", (int(df['Math'].min()), int(df['Math'].max())))
    print("Set (departments)   :", sorted(set(df["Department"])))
    counts = {}
    for d in df["Department"]:
        counts[d] = counts.get(d, 0) + 1                      # dictionary
    print("Dictionary          :", counts)
    print("Dict methods        : keys", list(counts.keys()), "| values", list(counts.values()))

    title("NUMPY: ARRAYS, INDEXING, SLICING, STATISTICS")
    a1 = df["Math"].to_numpy()
    a2 = df[SUBJECTS].to_numpy()
    print("1D shape", a1.shape, "| 2D shape", a2.shape)
    print("a1[0] =", a1[0], "| a1[:5] =", a1[:5])
    print("a2[:3, :] =\n", a2[:3, :])
    print(f"sum={a1.sum():.0f} mean={a1.mean():.2f} max={a1.max():.0f} "
          f"min={a1.min():.0f} std={a1.std():.2f}")

    title("PANDAS: SELECT, FILTER, SORT, GROUPBY, STATS")
    print("Selected rows/cols:\n", df.loc[0:4, ["Name", "Department", "Math"]].to_string(index=False))
    print("\nFilter (Average >= 75):", (df["Average"] >= 75).sum(), "students")
    print("\nSorted (top 3):\n", df.sort_values("Average", ascending=False).head(3)
          [["Name", "Average"]].to_string(index=False))
    print("\nGroupBy Department (mean):\n",
          df.groupby("Department")[SUBJECTS].mean().round(2).to_string())
    print("\nDescribe:\n", df[NUMERIC].describe().round(2).to_string())


def main():
    while True:
        df = load()
        title("STUDENT MANAGEMENT & ANALYTICS SYSTEM")
        print(" 1. Add new record          5. Delete a record")
        print(" 2. Display all records     6. Generate summary report")
        print(" 3. Search a record         7. Data analysis (insights)")
        print(" 4. Update a record         8. Concepts demo (DS / NumPy / Pandas)")
        print(" 0. Exit")
        choice = input("\nEnter choice: ").strip()
        try:
            if choice == "1":
                add_record(df)
            elif choice == "2":
                title(f"ALL RECORDS ({len(df)})")
                show(df)
            elif choice == "3":
                search_record(df)
            elif choice == "4":
                update_record(df)
            elif choice == "5":
                delete_record(df)
            elif choice == "6":
                summary_report(df)
            elif choice == "7":
                data_analysis(df)
            elif choice == "8":
                concepts_demo(df)
            elif choice == "0":
                print("Goodbye!")
                break
            else:
                print("Invalid choice. Enter a number from the menu.")
        except Exception as e:                       # never crash the menu
            print("Something went wrong:", e)


if __name__ == "__main__":
    main()