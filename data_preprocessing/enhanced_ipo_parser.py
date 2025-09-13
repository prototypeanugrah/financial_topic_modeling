import logging
import re
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

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
        self.supported_forms = ["S-1", "S-1/A", "424B3", "S-3", "F-1", "424B4", "424B5"]
        self.ipo_priority_forms = ["S-1", "S-1/A"]

    def parse_url_data(self, url_path: str):
        try:
            headers = {
                "User-Agent": "Sample Company Name AdminContact@company.com",
                "Host": "www.sec.gov",
            }

            data = requests.get(url_path, headers=headers, timeout=10)
            data.raise_for_status()

            # Split into documents
            documents = self.split_documents(data.text)
            # logger.info(f"Found {len(documents)} document(s) in file")

            # Select the appropriate document
            selected_doc = self.select_document(documents)

            # Parse the selected document
            return self.parse_document(selected_doc)

        except Exception as e:
            logger.error(f"Error parsing url {url_path}: {str(e)}")
            raise

    def parse_file(self, file_path: str) -> ParsedIPOData:
        """Main parsing method"""
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            # Split into documents
            documents = self.split_documents(content)
            # logger.info(f"Found {len(documents)} document(s) in file")

            # Select the appropriate document
            selected_doc = self.select_document(documents)

            # Parse the selected document
            return self.parse_document(selected_doc)

        except Exception as e:
            logger.error(f"Error parsing file {file_path}: {str(e)}")
            raise

    def split_documents(self, content: str) -> List[str]:
        """Split content into individual SEC documents"""
        # Split by SEC-DOCUMENT tags
        documents = re.split(r"<SEC-DOCUMENT[^>]*>", content)

        # Remove empty documents and the content before first document
        valid_documents = []
        for doc in documents[1:]:  # Skip content before first <SEC-DOCUMENT>
            if doc.strip() and "SEC-HEADER" in doc:
                valid_documents.append(doc)

        return valid_documents if valid_documents else [content]

    def get_form_type(self, document: str) -> Optional[str]:
        """Extract form type from document"""
        match = re.search(r"CONFORMED SUBMISSION TYPE:\s+([^\r\n]+)", document)
        if match:
            form_type = match.group(1).strip()
            return form_type
        return None

    def select_document(self, documents: List[str]) -> str:
        """Select the appropriate document based on priority rules"""
        if len(documents) == 1:
            return documents[0]

        # Look for IPO priority forms first (S-1, S-1/A)
        for form_type in self.ipo_priority_forms:
            for doc in documents:
                doc_form = self.get_form_type(doc)
                if doc_form and doc_form.strip() == form_type:
                    # logger.info(f"Selected first {form_type} document")
                    return doc

        # If no S-1 or S-1/A found, return the first supported document
        for doc in documents:
            doc_form = self.get_form_type(doc)
            if doc_form:
                for supported_form in self.supported_forms:
                    if doc_form.strip() == supported_form:
                        # logger.info(f"Selected {doc_form} document")
                        return doc

        # Fallback to first document
        logger.warning("No recognized form type found, using first document")
        return documents[0]

    def parse_document(self, document: str) -> ParsedIPOData:
        """Parse a single SEC document"""
        # Extract header and content sections
        header_match = re.search(r"<SEC-HEADER>(.*?)</SEC-HEADER>", document, re.DOTALL)

        if not header_match:
            raise ValueError("No SEC-HEADER found in document")

        header_content = header_match.group(1)

        # Parse header information
        company_info = self.parse_company_info(header_content)
        filing_info = self.parse_filing_info(header_content)

        # Create complete parsed data
        parsed_data = ParsedIPOData(
            company_info=company_info,
            filing_info=filing_info,
            raw_document_type=self.get_form_type(document),
        )

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

        # Form type
        match = re.search(r"CONFORMED SUBMISSION TYPE:\s+(.+)", header)
        if match:
            filing_info.form_type = match.group(1).strip()

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
