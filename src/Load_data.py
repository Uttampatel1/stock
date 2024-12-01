import requests
import pandas as pd
from bs4 import BeautifulSoup
from typing import Dict, List, TypedDict, Optional
from datetime import datetime
import json
import re
from typing import Dict, Union, Optional


class MetricsFormatter:
    @staticmethod
    def clean_number(value: str) -> str:
        """Remove extra whitespace, newlines, and format numbers"""
        value = ' '.join(value.split())
        value = re.sub(r'\s+([0-9,])', r'\1', value)
        return value.strip()

    @staticmethod
    def format_currency(value: str) -> Dict[str, Union[float, str]]:
        """Format currency values"""
        cleaned = value.replace('₹', '').strip()
        is_crore = 'Cr.' in cleaned
        cleaned = cleaned.replace('Cr.', '').strip()
        
        try:
            number = float(cleaned.replace(',', ''))
            if is_crore:
                formatted_value = f"₹{number:,.2f} Cr."
            else:
                formatted_value = f"₹{number:,.2f}"
            return {
                "value": number,
                "formatted": formatted_value,
                "denomination": "Crore" if is_crore else "Absolute"
            }
        except ValueError:
            return {
                "value": 0.0,
                "formatted": "₹0.00",
                "denomination": "Unknown"
            }

    @staticmethod
    def format_percentage(value: str) -> Dict[str, Union[float, str]]:
        """Format percentage values"""
        cleaned = value.replace('%', '').strip()
        try:
            number = float(cleaned)
            return {
                "value": number,
                "formatted": f"{number:.2f}%"
            }
        except ValueError:
            return {
                "value": 0.0,
                "formatted": "0.00%"
            }

    @staticmethod
    def format_high_low(value: str) -> Dict[str, Union[float, Dict[str, float], str]]:
        """Format high/low range values"""
        cleaned = value.replace('₹', '').strip()
        try:
            high, low = cleaned.split('/')
            high_val = float(high.strip().replace(',', ''))
            low_val = float(low.strip().replace(',', ''))
            return {
                "high": high_val,
                "low": low_val,
                "formatted": f"₹{high_val:,.2f} / ₹{low_val:,.2f}",
                "range": high_val - low_val
            }
        except (ValueError, IndexError):
            return {
                "high": 0.0,
                "low": 0.0,
                "formatted": "₹0.00 / ₹0.00",
                "range": 0.0
            }

class CompanyMetricsFormatter:
    def __init__(self):
        self.formatter = MetricsFormatter()

    def format_metrics(self, metrics: Dict[str, str]) -> Dict[str, Dict[str, Union[float, str, Dict]]]:
        """Format all company metrics"""
        formatted_metrics = {}

        metric_formatters = {
            "Market Cap": ("market_cap", self.formatter.format_currency),
            "Current Price": ("current_price", self.formatter.format_currency),
            "High / Low": ("high_low", self.formatter.format_high_low),
            "Stock P/E": ("stock_pe", lambda x: {
                "value": float(self.formatter.clean_number(x)),
                "formatted": f"{float(self.formatter.clean_number(x)):,.2f}"
            }),
            "Book Value": ("book_value", self.formatter.format_currency),
            "Dividend Yield": ("dividend_yield", self.formatter.format_percentage),
            "ROCE": ("roce", self.formatter.format_percentage),
            "ROE": ("roe", self.formatter.format_percentage),
            "Face Value": ("face_value", self.formatter.format_currency)
        }

        for original_key, (new_key, formatter) in metric_formatters.items():
            if original_key in metrics:
                formatted_metrics[new_key] = formatter(metrics[original_key])

        return formatted_metrics

# Update the process_company_metrics method in StockAnalyzer class
def process_company_metrics(self) -> None:
    """Process company metrics"""
    try:
        raw_metrics = {}
        ratios_section = self.soup.find('div', class_='company-ratios')
        
        if ratios_section:
            for item in ratios_section.find_all('li', class_='flex'):
                name_elem = item.find('span', class_='name')
                value_elem = item.find('span', class_='value')
                
                if name_elem and value_elem:
                    name = name_elem.text.strip()
                    value = value_elem.text.strip()
                    raw_metrics[name] = value

            # Format the metrics using CompanyMetricsFormatter
            formatter = CompanyMetricsFormatter()
            self.data["company_metrics"] = formatter.format_metrics(raw_metrics)
            
            if not self.data["company_metrics"]:
                print("Warning: No company metrics found")

        else:
            print("Warning: Company ratios section not found")
            self.data["company_metrics"] = {}

    except Exception as e:
        print(f"Error processing company metrics: {e}")
        self.data["company_metrics"] = {}

class StockAnalyzer:
    def __init__(self, symbol: str):
        self.symbol = symbol.upper()
        self.base_url = f"https://www.screener.in/company/{self.symbol}/consolidated/"
        self.soup = None
        self.data = {
            "company_metrics": {},
            "quarters_data": pd.DataFrame(),
            "profit_loss_data": pd.DataFrame(),
            "balance_sheet_data": pd.DataFrame(),
            "cash_flow_data": pd.DataFrame(),
            "ratios_data": pd.DataFrame(),
            "shareholding_data": pd.DataFrame(),
            "documents_data": pd.DataFrame(),
            "pros": [],
            "cons": []
        }

    def fetch_data(self) -> bool:
        """Fetch data from screener.in"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(self.base_url, headers=headers)
            response.raise_for_status()
            self.soup = BeautifulSoup(response.content, 'html.parser')
            print(f"Successfully fetched data for {self.symbol}")
            return True
        except requests.RequestException as e:
            print(f"Error fetching data: {e}")
            return False

    def _safe_extract_table(self, section_id: str) -> pd.DataFrame:
        """Safely extract table data"""
        try:
            section = self.soup.find('section', id=section_id)
            if not section:
                print(f"Warning: Section '{section_id}' not found")
                return pd.DataFrame()

            table = section.find('table')
            if not table:
                print(f"Warning: Table not found in section '{section_id}'")
                return pd.DataFrame()

            headers = []
            for th in table.find_all('th'):
                if th.text:
                    headers.append(th.text.strip())
                else:
                    headers.append("")

            rows = []
            for tr in table.find_all('tr')[1:]:  # Skip header row
                row = []
                for td in tr.find_all('td'):
                    row.append(td.text.strip() if td.text else "")
                if row:  # Only append non-empty rows
                    rows.append(row)

            if not headers or not rows:
                print(f"Warning: No data found in table for section '{section_id}'")
                return pd.DataFrame()

            return pd.DataFrame(rows, columns=headers)

        except Exception as e:
            print(f"Error processing {section_id} table: {e}")
            return pd.DataFrame()

    def process_quarters_data(self) -> None:
        """Process quarterly financial data"""
        self.data["quarters_data"] = self._safe_extract_table('quarters')

    def process_profit_loss_data(self) -> None:
        """Process profit and loss data"""
        self.data["profit_loss_data"] = self._safe_extract_table('profit-loss')

    def process_balance_sheet_data(self) -> None:
        """Process balance sheet data"""
        self.data["balance_sheet_data"] = self._safe_extract_table('balance-sheet')

    def process_cash_flow_data(self) -> None:
        """Process cash flow data"""
        self.data["cash_flow_data"] = self._safe_extract_table('cash-flow')

    def process_ratios_data(self) -> None:
        """Process financial ratios data"""
        self.data["ratios_data"] = self._safe_extract_table('ratios')

    def process_shareholding_data(self) -> None:
        """Process shareholding data"""
        self.data["shareholding_data"] = self._safe_extract_table('shareholding')

    def process_company_metrics(self) -> None:
        """Process company metrics"""
        try:
            raw_metrics = {}
            ratios_section = self.soup.find('div', class_='company-ratios')
            
            if ratios_section:
                for item in ratios_section.find_all('li', class_='flex'):
                    name_elem = item.find('span', class_='name')
                    value_elem = item.find('span', class_='value')
                    
                    if name_elem and value_elem:
                        name = name_elem.text.strip()
                        value = value_elem.text.strip()
                        raw_metrics[name] = value

                # Format the metrics using CompanyMetricsFormatter
                formatter = CompanyMetricsFormatter()
                self.data["company_metrics"] = formatter.format_metrics(raw_metrics)
                
                if not self.data["company_metrics"]:
                    print("Warning: No company metrics found")

            else:
                print("Warning: Company ratios section not found")
                self.data["company_metrics"] = {}

        except Exception as e:
            print(f"Error processing company metrics: {e}")
            self.data["company_metrics"] = {}


    def process_pros_cons(self) -> None:
        """Process pros and cons"""
        try:
            pros_section = self.soup.find('div', class_='pros')
            cons_section = self.soup.find('div', class_='cons')
            
            self.data["pros"] = [pro.text.strip() for pro in (pros_section.find_all('li') if pros_section else [])]
            self.data["cons"] = [con.text.strip() for con in (cons_section.find_all('li') if cons_section else [])]
            
        except Exception as e:
            print(f"Error processing pros and cons: {e}")
            self.data["pros"] = []
            self.data["cons"] = []

    def analyze(self) -> bool:
        """Perform complete analysis"""
        try:
            if not self.fetch_data():
                return False

            self.process_company_metrics()
            self.process_quarters_data()
            self.process_profit_loss_data()
            self.process_balance_sheet_data()
            self.process_cash_flow_data()
            self.process_ratios_data()
            self.process_shareholding_data()
            self.process_pros_cons()

            # Validate that we got at least some data
            has_data = False
            for key, value in self.data.items():
                if isinstance(value, pd.DataFrame) and not value.empty:
                    has_data = True
                    break
                elif isinstance(value, dict) and value:
                    has_data = True
                    break
                elif isinstance(value, list) and value:
                    has_data = True
                    break

            if not has_data:
                print(f"Warning: No data was successfully retrieved for {self.symbol}")
                return False

            print(f"Analysis completed for {self.symbol}")
            return True

        except Exception as e:
            print(f"Error during analysis: {e}")
            return False

    def save_analysis(self, base_filename: Optional[str] = None) -> None:
        """Save analysis to files"""
        if base_filename is None:
            base_filename = self.symbol.lower()

        # Save main data to JSON
        main_data = {
            "symbol": self.symbol,
            "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "company_metrics": self.data["company_metrics"],
            "pros": self.data["pros"],
            "cons": self.data["cons"]
        }
        
        try:
            with open(f"{base_filename}_metrics.json", 'w', encoding='utf-8') as f:
                json.dump(main_data, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving metrics JSON: {e}")

        # Save DataFrames to CSV
        for key in ["quarters_data", "profit_loss_data", "balance_sheet_data",
                   "cash_flow_data", "ratios_data", "shareholding_data"]:
            try:
                if not self.data[key].empty:
                    self.data[key].to_csv(f"{base_filename}_{key}.csv", index=False, encoding='utf-8')
            except Exception as e:
                print(f"Error saving {key} CSV: {e}")

def analyze_stock(symbol: str) -> Dict:
    """Analyze a stock and return the data"""
    analyzer = StockAnalyzer(symbol)
    success = analyzer.analyze()
    if success:
        analyzer.save_analysis()
    return analyzer.data

# Example usage
if __name__ == "__main__":
    stock_symbol = "TATAMOTORS"
    stock_data = analyze_stock(stock_symbol)
    
    print("\nCompany Metrics:")
    for key, value in stock_data["company_metrics"].items():
        print(f"{key}: {value}")
    
    print("\nPros:")
    for pro in stock_data["pros"]:
        print(f"- {pro}")
    
    print("\nCons:")
    for con in stock_data["cons"]:
        print(f"- {con}")
    
    # Display data frame summaries
    for key in ["quarters_data", "profit_loss_data", "balance_sheet_data",
                "cash_flow_data", "ratios_data", "shareholding_data"]:
        df = stock_data[key]
        if not df.empty:
            print(f"\n{key} Preview:")
            print(df.head())