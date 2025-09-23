import concurrent.futures
import sys
from pathlib import Path

import pandas as pd
from enhanced_ipo_parser import SECFilingParser
from tqdm import tqdm

tqdm.pandas()


parser_instance = SECFilingParser()


def get_form_type(url: str) -> str:
    # Determine if it's a URL or local file
    if url.startswith(("http://", "https://")):
        print(f"Parsing URL: {url}")
        parsed_data = parser_instance.parse_url_data(url)
    elif url.startswith("edgar/"):
        # add the base url
        url = "https://sec.gov/Archives/" + url
        parsed_data = parser_instance.parse_url_data(url)
    else:
        url = Path(url)
        if not url.exists():
            print(f"Error: File not found: {url}")
            sys.exit(1)
        print(f"Parsing file: {url}")
        parsed_data = parser_instance.parse_file(str(url))

    if parsed_data is None:
        print("No valid SEC filing found in the document.")
        sys.exit(1)

    # Always show form type
    form_type = parsed_data.filing_info.form_type or parsed_data.raw_document_type
    return form_type


def process_concurrent(
    df,
    column,
    func,
    max_workers=4,
):
    """Process DataFrame column concurrently"""
    urls = df[column].tolist()
    results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_url = {executor.submit(func, url): url for url in urls}

        # Process completed tasks with progress bar
        for future in tqdm(
            concurrent.futures.as_completed(future_to_url),
            total=len(urls),
            desc="Processing URLs",
        ):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                url = future_to_url[future]
                print(f"Error processing {url}: {e}")
                results.append(None)

    return results


def main():
    # Parameters
    # subset_rows = 100
    max_workers = 8

    # Read the data
    all_reports_v3 = pd.read_csv("data/rep180_v3_initial.csv")

    # keep only the records where filenames1 is not null
    all_reports_v3 = all_reports_v3[all_reports_v3["filenames1"].notna()].reset_index(
        drop=True
    )
    print(all_reports_v3.shape)
    print(f"Unique IPO report URLs: {all_reports_v3.filenames1.nunique()}")

    # Get the form type
    subset = all_reports_v3.copy()
    subset["form_type"] = process_concurrent(
        subset, "filenames1", get_form_type, max_workers=max_workers
    )

    # Get the formatted s1date
    subset["formatted_s1date"] = pd.to_datetime(subset["s1date"])

    df = subset[["CUSIP6", "filenames1", "form_type", "formatted_s1date"]]

    # for a CUSIP6, find the min formatted_s1date for S-1 and max formatted_s1date for S-1/A
    keeper = (
        df.groupby(["CUSIP6", "form_type"])["formatted_s1date"]
        .agg(min_date="min", max_date="max")
        .reset_index()
    )

    choice = {"S-1": "min_date", "S-1/A": "max_date"}

    keeper["keep_date"] = keeper.apply(
        lambda r: r[choice.get(r["form_type"], "min_date")],
        axis=1,
    )

    keepers = keeper[["CUSIP6", "form_type", "keep_date"]].rename(
        columns={"keep_date": "formatted_s1date"}
    )

    # Inner-join back to keep only the extreme rows
    filtered = df.merge(
        keepers,
        on=["CUSIP6", "form_type", "formatted_s1date"],
        how="inner",
    ).drop_duplicates()

    filtered.sort_values(
        by=["form_type", "formatted_s1date"],
        ascending=True,
        inplace=True,
    )

    filtered.to_csv("data/latest_ipo_s1_filings.csv", index=False)


if __name__ == "__main__":
    main()
