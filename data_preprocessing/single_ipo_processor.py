import argparse
import logging
from pathlib import Path
from typing import Optional

from enhanced_ipo_parser import ParsedIPOData, SECFilingParser

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Single IPO report processor")
    parser.add_argument(
        "--url",
        type=str,
        required=True,
        help="Single URL to process (e.g., 'edgar/data/1013880/0000912057-96-010403.txt')",
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        required=False,
        default="single_ipo_report",
        help="Output prefix for generated files",
    )
    parser.add_argument(
        "--raw_reports_dir",
        type=str,
        required=False,
        default="data/raw_reports_testing",
        help="Raw reports directory",
    )
    parser.add_argument(
        "--save_text",
        action="store_true",
        help="Save the full text content to a file",
    )
    parser.add_argument(
        "--save_metadata",
        action="store_true",
        help="Save metadata to CSV file",
    )
    return parser.parse_args()


class SingleIPOProcessor:
    """Process a single IPO report with the same functionality as batch processor"""

    def __init__(self, raw_reports_dir: str):
        self.base_url = "https://www.sec.gov/Archives/"

        # Create raw_reports directory
        self.raw_reports_dir = Path(raw_reports_dir)
        self.raw_reports_dir.mkdir(exist_ok=True)

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

    def process_single_url(self, url: str) -> Optional[ParsedIPOData]:
        """Process a single URL and return parsed data"""
        parser = SECFilingParser()

        try:
            # Construct full URL
            full_url = self.base_url + url
            logger.info(f"Processing URL: {full_url}")

            # Parse the document
            parsed_data = parser.parse_url_data(full_url)

            if parsed_data:
                logger.info("Successfully parsed document")
                return parsed_data
            else:
                logger.warning(
                    "Parser returned None - document may be empty or invalid"
                )
                return None

        except Exception as e:
            logger.error(f"Failed to process {url}: {str(e)}")
            return None

    def save_text_content(self, parsed_data: ParsedIPOData, url: str) -> Optional[Path]:
        """Save the full text content to a file"""
        if not parsed_data or not parsed_data.full_text_content:
            logger.warning("No text content to save")
            return None

        try:
            filename = self.extract_filename_from_url(url)
            text_file_path = self.raw_reports_dir / filename

            with open(text_file_path, "w", encoding="utf-8") as f:
                f.write(parsed_data.full_text_content)

            logger.info(f"Saved text content to: {text_file_path}")
            return text_file_path

        except Exception as e:
            logger.error(f"Failed to save text content: {str(e)}")
            return None

    def save_metadata(
        self, parsed_data: ParsedIPOData, url: str, output_prefix: str
    ) -> Optional[Path]:
        """Save metadata to CSV file"""
        if not parsed_data:
            logger.warning("No parsed data to save")
            return None

        try:
            parser = SECFilingParser()

            # Convert to metadata DataFrame
            df = parser.to_metadata_dataframe(parsed_data)

            # Add the raw URL
            df["raw_document_url"] = url

            # Save to CSV
            output_file = f"{output_prefix}.csv"
            df.to_csv(output_file, index=False, encoding="utf-8")

            logger.info(f"Saved metadata to: {output_file}")
            return Path(output_file)

        except Exception as e:
            logger.error(f"Failed to save metadata: {str(e)}")
            return None

    def print_summary(self, parsed_data: Optional[ParsedIPOData], url: str):
        """Print a summary of the parsed data"""
        print("\n" + "=" * 60)
        print("PROCESSING SUMMARY")
        print("=" * 60)
        print(f"URL: {url}")
        print(f"Full URL: {self.base_url + url}")

        if parsed_data:
            print("\n✅ SUCCESS: Document parsed successfully")

            # Company Info
            if parsed_data.company_info:
                print("\n📊 COMPANY INFORMATION:")
                print(
                    f"  Company Name: {parsed_data.company_info.company_name or 'N/A'}"
                )
                print(f"  CIK: {parsed_data.company_info.cik or 'N/A'}")
                print(f"  Ticker: {parsed_data.company_info.ticker_symbol or 'N/A'}")
                print(f"  SIC Code: {parsed_data.company_info.sic_code or 'N/A'}")
                print(
                    f"  SIC Description: {parsed_data.company_info.sic_description or 'N/A'}"
                )
                print(
                    f"  State of Incorporation: {parsed_data.company_info.state_of_incorporation or 'N/A'}"
                )

            # Filing Info
            if parsed_data.filing_info:
                print("\n�� FILING INFORMATION:")
                print(f"  Form Type: {parsed_data.filing_info.form_type or 'N/A'}")
                print(f"  Filing Date: {parsed_data.filing_info.filing_date or 'N/A'}")
                print(
                    f"  Accession Number: {parsed_data.filing_info.accession_number or 'N/A'}"
                )
                print(
                    f"  SEC File Number: {parsed_data.filing_info.sec_file_number or 'N/A'}"
                )

            # Text Content
            if parsed_data.full_text_content:
                text_length = len(parsed_data.full_text_content)
                print(f"\n📝 TEXT CONTENT: {text_length:,} characters")
            else:
                print("\n⚠️  No text content extracted")

        else:
            print("\n❌ FAILED: Document could not be parsed")

        print("=" * 60)

    def run(
        self,
        url: str,
        output_prefix: str,
        save_text: bool = True,
        save_metadata: bool = True,
    ):
        """Main method to process a single URL"""
        try:
            # Process the URL
            parsed_data = self.process_single_url(url)

            # Print summary
            self.print_summary(parsed_data, url)

            # Save files if requested and data is available
            saved_files = []

            if save_text and parsed_data:
                text_file = self.save_text_content(parsed_data, url)
                if text_file:
                    saved_files.append(text_file)

            if save_metadata and parsed_data:
                metadata_file = self.save_metadata(parsed_data, url, output_prefix)
                if metadata_file:
                    saved_files.append(metadata_file)

            # Print saved files
            if saved_files:
                print("\n💾 SAVED FILES:")
                for file_path in saved_files:
                    print(f"  - {file_path}")

            return parsed_data

        except Exception as e:
            logger.error(f"Single URL processing failed: {str(e)}")
            raise


def main():
    """Main function to run single URL processing"""

    # Parse arguments
    args = parse_args()
    url = args.url
    output_prefix = args.output_prefix
    raw_reports_dir = args.raw_reports_dir
    save_text = args.save_text
    save_metadata = args.save_metadata

    # If no save options specified, save both by default
    if not save_text and not save_metadata:
        save_text = True
        save_metadata = True

    # Create processor
    processor = SingleIPOProcessor(raw_reports_dir=raw_reports_dir)

    # Run processing
    processor.run(url, output_prefix, save_text, save_metadata)


if __name__ == "__main__":
    main()
