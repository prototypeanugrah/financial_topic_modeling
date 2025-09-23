import logging
import re
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import aiohttp
import pandas as pd
import requests
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CompanyInfo:
    """Company basic information"""

    company_name: Optional[str] = None
    cik: Optional[str] = None
    ticker_symbol: Optional[str] = None
    sic_code: Optional[str] = None
    sic_description: Optional[str] = None
    irs_number: Optional[str] = None
    state_of_incorporation: Optional[str] = None


@dataclass
class FilingInfo:
    """SEC filing metadata"""

    accession_number: Optional[str] = None
    form_type: Optional[str] = None
    filing_date: Optional[str] = None
    acceptance_datetime: Optional[str] = None
    sec_file_number: Optional[str] = None
    film_number: Optional[str] = None
    sec_act: Optional[str] = None


@dataclass
class ParsedIPOData:
    """Complete parsed IPO data structure"""

    company_info: CompanyInfo
    filing_info: FilingInfo
    raw_document_type: Optional[str] = None
    full_text_content: Optional[str] = None


class SECFilingParser:
    """Enhanced SEC filing parser with multi-document support"""

    def __init__(self):
        self.ipo_priority_forms = ["S-1", "S-1/A"]

    async def parse_url_data_async(self, url_path: str) -> Optional[ParsedIPOData]:
        try:
            headers = {
                "User-Agent": "Sample Company Name AdminContact@company.com",
                "Host": "www.sec.gov",
            }
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url_path, headers=headers, timeout=10
                ) as response:
                    response.raise_for_status()
                    content = await response.text()
                    documents = self.split_documents(content)
                    selected_doc, form_type, form_text = self.select_document(documents)
                    return self.parse_document(selected_doc, form_type, form_text)
        except Exception as e:
            logger.error(f"Error parsing url {url_path}: {str(e)}")
            raise

    def parse_url_data(self, url_path: str) -> Optional[ParsedIPOData]:
        try:
            headers = {
                "User-Agent": "Sample Company Name AdminContact@company.com",
                "Host": "www.sec.gov",
            }

            data = requests.get(url_path, headers=headers, timeout=15)
            data.raise_for_status()

            # Split into documents
            documents = self.split_documents(data.text)
            # logger.info(f"Found {len(documents)} document(s) in file")

            # Select the appropriate document
            selected_doc, form_type, form_text = self.select_document(documents)

            # Parse the selected document
            return self.parse_document(selected_doc, form_type, form_text)

        except ValueError as exc:
            logger.info("No target filing found in %s: %s", url_path, exc)
            return None

        except Exception as e:
            logger.error(f"Error parsing url {url_path}: {str(e)}")
            raise

    def parse_file(self, file_path: str) -> Optional[ParsedIPOData]:
        """Main parsing method"""
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            # Split into documents
            documents = self.split_documents(content)
            # logger.info(f"Found {len(documents)} document(s) in file")

            # Select the appropriate document
            selected_doc, form_type, form_text = self.select_document(documents)

            # Parse the selected document
            return self.parse_document(selected_doc, form_type, form_text)

        except ValueError as exc:
            logger.info("No target filing found in %s: %s", file_path, exc)
            return None

        except Exception as e:
            logger.error(f"Error parsing file {file_path}: {str(e)}")
            raise

    def split_documents(self, content: str) -> List[str]:
        """Split content into individual SEC documents"""

        content = content or ""

        try:
            sec_documents = re.findall(
                r"<SEC-DOCUMENT[^>]*>.*?</SEC-DOCUMENT>",
                content,
                flags=re.DOTALL | re.IGNORECASE,
            )
        except re.error as exc:
            logger.error("Failed to locate SEC-DOCUMENT blocks: %s", exc)
            sec_documents = []

        cleaned_documents: List[str] = []

        for sec_doc in sec_documents:
            inner_doc = re.sub(
                r"^<SEC-DOCUMENT[^>]*>", "", sec_doc, flags=re.IGNORECASE
            )
            inner_doc = re.sub(
                r"</SEC-DOCUMENT>\s*$", "", inner_doc, flags=re.IGNORECASE
            )
            if inner_doc.strip():
                cleaned_documents.append(inner_doc)

        if cleaned_documents:
            return cleaned_documents

        try:
            document_sections = re.findall(
                r"<DOCUMENT>.*?</DOCUMENT>",
                content,
                flags=re.DOTALL | re.IGNORECASE,
            )
        except re.error as exc:
            logger.error("Failed to locate DOCUMENT blocks: %s", exc)
            document_sections = []

        if document_sections:
            return document_sections

        stripped_content = content.strip()
        return [stripped_content] if stripped_content else []

    def get_form_type(self, document: str) -> Optional[str]:
        """Extract form type from document using <TYPE> tag"""
        # First try to find <TYPE> tag in document sections
        sections = re.findall(
            r"(<DOCUMENT>.*?</DOCUMENT>)",
            document,
            flags=re.DOTALL | re.IGNORECASE,
        )

        for section in sections:
            type_match = re.search(
                r"<TYPE>\s*([^\n\r<]+)", section, flags=re.IGNORECASE
            )
            if type_match:
                form_type = type_match.group(1).strip()
                return form_type

        return None

    def select_document(
        self, documents: List[str]
    ) -> Tuple[str, Optional[str], Optional[str]]:
        """Select SEC document containing target filing type"""

        for document in documents:
            section_types, section_texts = self._extract_target_sections(document)
            if section_texts:
                combined_types = (
                    "|".join(section_types)
                    if len(section_types) > 1
                    else section_types[0]
                )
                combined_text = "\n\n".join(section_texts)
                return document, combined_types, combined_text

        if documents:
            logger.info(
                "No S-1 or S-1/A section found; returning document without target form"
            )
            return documents[0], None, None

        raise ValueError("No documents found in filing")

    def _extract_target_sections(self, document: str) -> Tuple[List[str], List[str]]:
        """Return ordered lists of form types and text blocks for target forms"""

        try:
            sections = re.findall(
                r"(<DOCUMENT>.*?</DOCUMENT>)",
                document,
                flags=re.DOTALL | re.IGNORECASE,
            )
        except re.error as exc:
            logger.error("Failed to split document sections: %s", exc)
            sections = []

        if not sections:
            stripped = document.strip()
            sections = [stripped] if stripped else []

        target_map = {form.upper(): form for form in self.ipo_priority_forms}
        matched_sections: List[Tuple[int, int, str, str]] = []

        for position, section in enumerate(sections):
            type_match = re.search(
                r"<TYPE>\s*([^\n\r<]+)", section, flags=re.IGNORECASE
            )
            if not type_match:
                continue

            raw_type = type_match.group(1).strip()
            canonical = target_map.get(raw_type.upper())
            if not canonical:
                continue

            text_match = re.search(
                r"<TEXT>(.*?)</TEXT>", section, flags=re.DOTALL | re.IGNORECASE
            )
            if text_match:
                section_text = text_match.group(1)
            else:
                section_text = section[type_match.end() :]
                section_text = re.sub(
                    r"</DOCUMENT>\s*$",
                    "",
                    section_text,
                    flags=re.DOTALL | re.IGNORECASE,
                )
                section_text = section_text.strip()
                if not section_text:
                    continue

            priority_index = self.ipo_priority_forms.index(canonical)
            matched_sections.append((priority_index, position, canonical, section_text))

        if not matched_sections:
            return [], []

        # Sort by priority (S-1, then S-1/A) while preserving natural order per group
        matched_sections.sort(key=lambda item: (item[0], item[1]))

        section_types = [item[2] for item in matched_sections]
        section_texts = [item[3] for item in matched_sections]
        return section_types, section_texts

    def parse_document(
        self,
        document: str,
        selected_form_type: Optional[str],
        selected_form_text: Optional[str],
    ) -> ParsedIPOData:
        """Parse a single SEC document"""
        # Extract header and content sections (header can be absent)
        header_match = re.search(
            r"<SEC-HEADER>(.*?)</SEC-HEADER>", document, re.DOTALL | re.IGNORECASE
        )

        header_content = header_match.group(1) if header_match else ""

        # Parse header information
        company_info = self.parse_company_info(header_content)
        filing_info = self.parse_filing_info(header_content)

        # Create complete parsed data
        parsed_data = ParsedIPOData(
            company_info=company_info,
            filing_info=filing_info,
            raw_document_type=(selected_form_type or self.get_form_type(document)),
        )

        if selected_form_text:
            cleaned_text = self.extract_full_text(selected_form_text)
            parsed_data.full_text_content = cleaned_text

        return parsed_data

    def parse_company_info(self, header: str) -> CompanyInfo:
        """Parse company information from SEC header"""
        company_info = CompanyInfo()

        # Company name
        match = re.search(r"COMPANY CONFORMED NAME:\s+(.+)", header)
        if match:
            company_info.company_name = match.group(1).strip()

        # CIK
        match = re.search(r"CENTRAL INDEX KEY:\s+(\d+)", header)
        if match:
            company_info.cik = match.group(1)

        # SIC code and description
        match = re.search(r"STANDARD INDUSTRIAL CLASSIFICATION:\s+(.+)", header)
        if match:
            sic_text = match.group(1).strip()
            sic_code_match = re.search(r"\[(\d+)\]", sic_text)
            if sic_code_match:
                company_info.sic_code = sic_code_match.group(1)
                company_info.sic_description = re.sub(r"\s*\[\d+\]", "", sic_text)

        # IRS number
        match = re.search(r"IRS NUMBER:\s+(\d+)", header)
        if match:
            company_info.irs_number = match.group(1)

        # State of incorporation
        match = re.search(r"STATE OF INCORPORATION:\s+(.+)", header)
        if match:
            company_info.state_of_incorporation = match.group(1).strip()

        return company_info

    def parse_filing_info(self, header: str) -> FilingInfo:
        """Parse filing information from SEC header"""
        filing_info = FilingInfo()

        # Accession number
        match = re.search(r"ACCESSION NUMBER:\s+(.+)", header)
        if match:
            filing_info.accession_number = match.group(1).strip()

        # # Form type
        # match = re.search(r"CONFORMED SUBMISSION TYPE:\s+(.+)", header)
        # if match:
        #     filing_info.form_type = match.group(1).strip()

        # Filing date
        match = re.search(r"FILED AS OF DATE:\s+(\d{8})", header)
        if match:
            filing_info.filing_date = match.group(1)

        # Acceptance datetime
        match = re.search(r"ACCEPTANCE-DATETIME:\s+(\d+)", header)
        if match:
            filing_info.acceptance_datetime = match.group(1)

        # SEC file number
        match = re.search(r"SEC FILE NUMBER:\s+(.+)", header)
        if match:
            filing_info.sec_file_number = match.group(1).strip()

        # Film number
        match = re.search(r"FILM NUMBER:\s+(\d+)", header)
        if match:
            filing_info.film_number = match.group(1)

        # SEC act
        match = re.search(r"SEC ACT:\s+(.+)", header)
        if match:
            filing_info.sec_act = match.group(1).strip()

        return filing_info

    def extract_full_text(self, html_content: str) -> str:
        """Extract all text content from HTML, removing tags but preserving structure"""
        if not html_content.strip():
            return ""

        try:
            # Parse with BeautifulSoup
            soup = BeautifulSoup(html_content, "html.parser")

            # Remove script and style elements completely
            for script in soup(["script", "style"]):
                script.decompose()

            # Get all text first
            full_text = soup.get_text()

            # Clean up the text
            # Replace HTML entities
            import html

            full_text = html.unescape(full_text)

            # Normalize whitespace but preserve paragraph breaks
            lines = full_text.split("\n")
            cleaned_lines = []

            for line in lines:
                # Clean up each line
                cleaned_line = re.sub(r"\s+", " ", line.strip())
                if cleaned_line:
                    cleaned_lines.append(cleaned_line)

            # Join lines with proper spacing
            full_text = "\n\n".join(cleaned_lines)

            # Remove excessive line breaks (more than 2 consecutive)
            full_text = re.sub(r"\n{3,}", "\n\n", full_text)

            return full_text

        except Exception as e:
            logger.warning(f"Error extracting full text: {str(e)}")
            # Fallback: just strip HTML tags with regex
            text = re.sub(r"<[^>]+>", "", html_content)
            text = html.unescape(text) if "html" in globals() else text
            text = re.sub(r"\s+", " ", text).strip()
            return text

    def to_dict(self, parsed_data: ParsedIPOData) -> Dict:
        """Convert parsed data to dictionary"""
        return asdict(parsed_data)

    def to_metadata_dataframe(self, parsed_data: ParsedIPOData) -> pd.DataFrame:
        """Convert parsed data to flat DataFrame excluding full text content"""
        data_dict = self.to_dict(parsed_data)

        # Remove full text content and text sections to keep only metadata
        data_dict.pop("full_text_content", None)
        data_dict.pop("text_sections", None)

        # Flatten nested structures
        flat_dict = {}
        for key, value in data_dict.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    flat_dict[f"{key}_{sub_key}"] = sub_value
            elif isinstance(value, list):
                # Convert lists to strings for CSV compatibility
                flat_dict[key] = "; ".join(str(item) for item in value)
            else:
                flat_dict[key] = value

        # Define columns to keep
        columns_to_keep = [
            "company_info_company_name",
            "company_info_cik",
            "company_info_ticker_symbol",
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

        # Filter to only keep specified columns
        filtered_dict = {}
        for col in columns_to_keep:
            filtered_dict[col] = flat_dict.get(col, None)

        # Add raw_document_url field if not already present
        if "raw_document_url" not in filtered_dict:
            filtered_dict["raw_document_url"] = None

        return pd.DataFrame([filtered_dict])


def main():
    """Main function to run the script standalone"""
    import argparse
    import sys
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Parse SEC filing documents and extract form type and metadata"
    )
    parser.add_argument(
        "--file_path", help="Path to the SEC filing document (local file or URL)"
    )
    parser.add_argument(
        "--show-all",
        action="store_true",
        help="Show all parsed metadata, not just form type",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Initialize parser
    parser_instance = SECFilingParser()

    try:
        # Determine if it's a URL or local file
        if args.file_path.startswith(("http://", "https://")):
            print(f"Parsing URL: {args.file_path}")
            parsed_data = parser_instance.parse_url_data(args.file_path)
        elif args.file_path.startswith("edgar/"):
            # add the base url
            args.file_path = "https://sec.gov/Archives/" + args.file_path
            print(f"Parsing EDGAR URL: {args.file_path}")
            parsed_data = parser_instance.parse_url_data(args.file_path)
        else:
            file_path = Path(args.file_path)
            if not file_path.exists():
                print(f"Error: File not found: {args.file_path}")
                sys.exit(1)
            print(f"Parsing file: {args.file_path}")
            parsed_data = parser_instance.parse_file(str(file_path))

        if parsed_data is None:
            print("No valid SEC filing found in the document.")
            sys.exit(1)

        # Display results
        print("\n" + "=" * 60)
        print("PARSING RESULTS")
        print("=" * 60)

        # Always show form type
        form_type = parsed_data.filing_info.form_type or parsed_data.raw_document_type
        print(f"Form Type: {form_type}")

        if args.show_all:
            print("\n" + "-" * 40)
            print("COMPANY INFORMATION")
            print("-" * 40)
            company = parsed_data.company_info
            print(f"Company Name: {company.company_name}")
            print(f"CIK: {company.cik}")
            print(f"Ticker Symbol: {company.ticker_symbol}")
            print(f"SIC Code: {company.sic_code}")
            print(f"SIC Description: {company.sic_description}")
            print(f"IRS Number: {company.irs_number}")
            print(f"State of Incorporation: {company.state_of_incorporation}")

            print("\n" + "-" * 40)
            print("FILING INFORMATION")
            print("-" * 40)
            filing = parsed_data.filing_info
            print(f"Accession Number: {filing.accession_number}")
            print(f"Form Type: {filing.form_type}")
            print(f"Filing Date: {filing.filing_date}")
            print(f"Acceptance DateTime: {filing.acceptance_datetime}")
            print(f"SEC File Number: {filing.sec_file_number}")
            print(f"Film Number: {filing.film_number}")
            print(f"SEC Act: {filing.sec_act}")

            print("\n" + "-" * 40)
            print("DOCUMENT INFORMATION")
            print("-" * 40)
            print(f"Raw Document Type: {parsed_data.raw_document_type}")
            if parsed_data.full_text_content:
                text_length = len(parsed_data.full_text_content)
                print(f"Full Text Content Length: {text_length:,} characters")
                print("Full Text Preview (first 200 chars):")
                print(f"'{parsed_data.full_text_content[:200]}...'")
            else:
                print("Full Text Content: Not extracted")

        print("\n" + "=" * 60)
        print("Parsing completed successfully!")

    except Exception as e:
        print(f"Error parsing document: {str(e)}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
