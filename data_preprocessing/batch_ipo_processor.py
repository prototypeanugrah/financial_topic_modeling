import argparse
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from enhanced_ipo_parser import ParsedIPOData, SECFilingParser
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Batch IPO processor")
    parser.add_argument(
        "--input_csv",
        type=str,
        required=False,
        default="data/rep180_v2_initial(in).csv",
        help="Input CSV file",
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        required=False,
        default="ipo_reports_metadata",
        help="Output prefix",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        required=False,
        default=100,
        help="Batch size",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        required=False,
        default=8,
        help="Number of processes",
    )
    parser.add_argument(
        "--raw_reports_dir",
        type=str,
        required=False,
        default="data/raw_reports",
        help="Raw reports directory",
    )
    return parser.parse_args()


class BatchIPOProcessor:
    """Process multiple IPO reports in parallel with batch processing"""

    def __init__(
        self,
        batch_size: int,
        num_processes: int,
        raw_reports_dir: str,
    ):
        self.batch_size = batch_size
        self.num_processes = num_processes
        self.base_url = "https://www.sec.gov/Archives/"

        # Create raw_reports directory
        self.raw_reports_dir = Path(raw_reports_dir)
        self.raw_reports_dir.mkdir(exist_ok=True)

    def load_unique_urls(self, csv_path: str) -> List[str]:
        """Load unique URLs from the input CSV file"""
        try:
            # Read CSV with latin-1 encoding to handle special characters
            df = pd.read_csv(csv_path, encoding="latin-1")
            unique_urls = df["url"].drop_duplicates().tolist()
            logger.info(
                f"Loaded {len(unique_urls)} unique URLs from {len(df)} total rows"
            )
            return unique_urls
        except Exception as e:
            logger.error(f"Error reading CSV file {csv_path}: {str(e)}")
            raise

    def extract_filename_from_url(self, url: str) -> str:
        """Extract filename from URL path in format: cik_accession.txt"""
        # URL format: edgar/data/{cik}/{accession}.txt
        # Extract CIK and accession number from path
        parts = url.split("/")

        # Find CIK (the part after 'data')
        cik = None
        accession = None

        for i, part in enumerate(parts):
            if part == "data" and i + 1 < len(parts):
                cik = parts[i + 1]
                break

        # Find the .txt file (accession number)
        for part in reversed(parts):
            if ".txt" in part:
                accession = part.replace(".txt", "")
                break

        # Create filename in format: cik_accession.txt
        if cik and accession:
            return f"{cik}_{accession}.txt"

        # Fallback to original format if parsing fails
        for part in reversed(parts):
            if ".txt" in part:
                return part
        return parts[-1] if parts else "unknown.txt"

    def process_batch(self, batch_urls: List[str]) -> List[Tuple[str, ParsedIPOData]]:
        """Process a batch of URLs and return parsed data with URLs"""
        parser = SECFilingParser()
        results = []

        for url in batch_urls:
            try:
                # Construct full URL
                full_url = self.base_url + url

                # Parse the document
                parsed_data = parser.parse_url_data(full_url)

                # Save individual text file
                if parsed_data and parsed_data.full_text_content:
                    filename = self.extract_filename_from_url(url)
                    text_file_path = self.raw_reports_dir / filename

                    with open(text_file_path, "w", encoding="utf-8") as f:
                        f.write(parsed_data.full_text_content)

                # Add the raw URL to the results
                results.append((url, parsed_data))

            except Exception as e:
                logger.warning(f"Failed to process {url}: {str(e)}")
                # Add None result to maintain consistency
                results.append((url, None))

        return results

    def create_batches(self, urls: List[str]) -> List[List[str]]:
        """Create batches of URLs for processing"""
        batches = []
        for i in range(0, len(urls), self.batch_size):
            batch = urls[i : i + self.batch_size]
            batches.append(batch)
        return batches

    def process_all_urls(self, urls: List[str], output_prefix: str) -> pd.DataFrame:
        """Process all URLs using multiprocessing and return consolidated DataFrame"""

        # Create batches
        batches = self.create_batches(urls)
        # logger.info(
        #     f"Created {len(batches)} batches of up to {self.batch_size} URLs each"
        # )

        # Process batches in parallel
        all_results = []

        with ProcessPoolExecutor(max_workers=self.num_processes) as executor:
            # Submit all batches
            future_to_batch = {
                executor.submit(self.process_batch, batch): i
                for i, batch in enumerate(batches)
            }

            # Collect results with progress bar
            with tqdm(total=len(batches), desc="Processing batches") as pbar:
                for future in as_completed(future_to_batch):
                    batch_idx = future_to_batch[future]
                    try:
                        batch_results = future.result()
                        all_results.extend(batch_results)
                        pbar.set_description(
                            f"Processed batch {batch_idx + 1}/{len(batches)}"
                        )
                        pbar.update(1)
                    except Exception as e:
                        logger.error(f"Batch {batch_idx} failed: {str(e)}")
                        pbar.update(1)

        # Convert results to DataFrame
        return self.create_consolidated_dataframe(all_results, output_prefix)

    def create_consolidated_dataframe(
        self, results: List[Tuple[str, ParsedIPOData]], output_prefix: str
    ) -> pd.DataFrame:
        """Create a consolidated DataFrame from all parsing results"""

        parser = SECFilingParser()
        consolidated_data = []

        for url, parsed_data in results:
            if parsed_data is not None:
                # Convert to metadata dictionary
                df = parser.to_metadata_dataframe(parsed_data)

                # Add the raw URL
                df["raw_document_url"] = url

                # Convert to dictionary for easy manipulation
                row_dict = df.iloc[0].to_dict()
                consolidated_data.append(row_dict)
            else:
                # Create empty row with URL for failed parsing
                empty_row = {
                    col: None
                    for col in [
                        "company_info_company_name",
                        "company_info_cik",
                        "company_info_ticker_symbol",
                        "company_info_exchange",
                        "company_info_sic_code",
                        "company_info_sic_description",
                        "company_info_irs_number",
                        "company_info_state_of_incorporation",
                        "filing_info_accession_number",
                        "filing_info_form_type",
                        "filing_info_filing_date",
                        "filing_info_sec_file_number",
                        "filing_info_film_number",
                        "raw_document_url",
                    ]
                }
                empty_row["raw_document_url"] = url
                consolidated_data.append(empty_row)

        # Create final DataFrame
        final_df = pd.DataFrame(consolidated_data)

        # Save consolidated CSV
        output_file = f"{output_prefix}.csv"
        final_df.to_csv(output_file, index=False, encoding="utf-8")

        return final_df

    def run(self, input_csv: str, output_prefix: str):
        """Main method to run the batch processing"""
        try:
            # Load unique URLs
            urls = self.load_unique_urls(input_csv)

            # Process all URLs
            final_df = self.process_all_urls(urls, output_prefix)

            # Print summary
            successful_parses = final_df["company_info_company_name"].notna().sum()
            logger.info(
                f"Processing complete: {successful_parses}/{len(final_df)} URLs parsed successfully"
            )

            return final_df

        except Exception as e:
            logger.error(f"Batch processing failed: {str(e)}")
            raise


def main():
    """Main function to run batch processing"""

    # Configuration
    args = parse_args()
    input_csv = args.input_csv
    output_prefix = args.output_prefix
    batch_size = args.batch_size
    num_processes = args.num_processes
    raw_reports_dir = args.raw_reports_dir
    # Create processor
    processor = BatchIPOProcessor(
        batch_size=batch_size,
        num_processes=num_processes,
        raw_reports_dir=raw_reports_dir,
    )

    # Run processing
    processor.run(input_csv, output_prefix)


if __name__ == "__main__":
    main()
