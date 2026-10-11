import re
import pandas as pd


def is_valid_email(value):

    pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'

    return bool(re.match(pattern, str(value)))


def is_valid_phone(value):

    pattern = r'^[6-9]\d{9}$'

    return bool(re.match(pattern, str(value)))


def is_valid_pin(value):

    pattern = r'^\d{6}$'

    return bool(re.match(pattern, str(value)))


def detect_format_problems(df):

    problems = []

    for column in df.columns:

        column_lower = column.lower()

        for index, value in df[column].items():

            if pd.isna(value):
                continue

            valid = True
            format_type = ""

            if "email" in column_lower:

                format_type = "Email"
                valid = is_valid_email(value)

            elif "phone" in column_lower or "mobile" in column_lower:

                format_type = "Phone"
                valid = is_valid_phone(value)

            elif "pin" in column_lower or "pincode" in column_lower:

                format_type = "PIN Code"
                valid = is_valid_pin(value)

            if format_type and not valid:

                problems.append({
                    "Row": index + 1,
                    "Column": column,
                    "Value": value,
                    "Format": format_type,
                    "Status": "Invalid Format"
                })

    return pd.DataFrame(problems)