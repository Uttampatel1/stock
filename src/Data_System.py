import requests
import pandas as pd
from bs4 import BeautifulSoup
from typing import Dict, List, TypedDict, Optional
from datetime import datetime
import json

class StockMetrics(TypedDict):
    quarterly_performance: Dict
    financial_ratios: Dict
    shareholding_pattern: Dict
    cash_flows: Dict
    credit_ratings: List
    pros: List[str]
    cons: List[str]

class StockAnalyzer:
    def __init__(self, symbol: str):
        self.symbol = symbol.upper()
        self.base_url = f"https://www.screener.in/company/{self.symbol}/consolidated/"
        self.data: StockMetrics = {
            "quarterly_performance": {},
            "financial_ratios": {},
            "shareholding_pattern": {},
            "cash_flows": {},
            "credit_ratings": [],
            "pros": [],
            "cons": []
        }
        
    def fetch_data(self) -> None:
        """Fetch data from screener.in"""
        try:
            response = requests.get(self.base_url)
            response.raise_for_status()
            self.soup = BeautifulSoup(response.content, 'html.parser')
            print(f"Successfully fetched data for {self.symbol}")
        except requests.RequestException as e:
            print(f"Error fetching data: {e}")
            raise
            
    def process_pros_cons(self) -> None:
        """Extract pros and cons"""
        # Find the Pros section
        pros_section = self.soup.find('div', class_='pros')
        pros_list = pros_section.find_all('li') if pros_section else []
        self.data["pros"] = [pro.text.strip() for pro in pros_list]

        # Find the Cons section
        cons_section = self.soup.find('div', class_='cons')
        cons_list = cons_section.find_all('li') if cons_section else []
        self.data["cons"] = [con.text.strip() for con in cons_list]
        
    def process_quarterly_data(self) -> None:
        """Process quarterly financial data"""
        quarters_section = self.soup.find('section', id='quarters')
        if not quarters_section:
            return
            
        table = quarters_section.find('table')
        if not table:
            return
            
        # Extract headers and data
        headers = [th.text.strip() for th in table.find_all('th')]
        rows = []
        for tr in table.find_all('tr')[1:]:
            cells = [td.text.strip() for td in tr.find_all('td')]
            if cells:
                rows.append(cells)
                
        df = pd.DataFrame(rows, columns=headers)
        
        # Convert to dictionary format
        for column in df.columns[1:]:  # Skip first column which is row labels
            quarter_data = {
                "Sales": self._convert_value(df.loc[0, column]),
                "Expenses": self._convert_value(df.loc[1, column]),
                "Operating_Profit": self._convert_value(df.loc[2, column]),
                "OPM": self._convert_value(df.loc[3, column].rstrip('%')),
                "Other_Income": self._convert_value(df.loc[4, column])
            }
            self.data["quarterly_performance"][column] = quarter_data
            
    def process_financial_ratios(self) -> None:
        """Process financial ratios"""
        ratios_section = self.soup.find('section', id='ratios')
        if not ratios_section:
            return
            
        table = ratios_section.find('table', class_='data-table')
        if not table:
            return
            
        headers = [th.text.strip() for th in table.find_all('th')]
        rows = []
        for tr in table.find_all('tr')[1:]:
            cells = [td.text.strip() for td in tr.find_all('td')]
            if cells:
                rows.append(cells)
                
        df = pd.DataFrame(rows, columns=headers)
        
        # Convert to dictionary format
        for column in df.columns[1:]:
            ratios = {
                "Debtor_Days": self._convert_value(df.loc[0, column]),
                "Inventory_Days": self._convert_value(df.loc[1, column]),
                "Days_Payable": self._convert_value(df.loc[2, column]),
                "Cash_Conversion_Cycle": self._convert_value(df.loc[3, column]),
                "Working_Capital_Days": self._convert_value(df.loc[4, column]),
                "ROCE": self._convert_value(df.loc[5, column].rstrip('%'))
            }
            self.data["financial_ratios"][column] = ratios
            
    def process_shareholding(self) -> None:
        """Process shareholding pattern"""
        shareholding_section = self.soup.find('section', id='shareholding')
        if not shareholding_section:
            return
            
        table = shareholding_section.find('table', class_='data-table')
        if not table:
            return
            
        headers = [th.text.strip() for th in table.find_all('th')]
        rows = []
        for tr in table.find_all('tr')[1:]:
            cells = [td.text.strip() for td in tr.find_all('td')]
            if cells:
                rows.append(cells)
                
        df = pd.DataFrame(rows, columns=headers)
        
        # Convert to dictionary format
        for column in df.columns[1:]:
            shareholding = {
                "Promoters": self._convert_value(df.loc[0, column].rstrip('%')),
                "FIIs": self._convert_value(df.loc[1, column].rstrip('%')),
                "DIIs": self._convert_value(df.loc[2, column].rstrip('%')),
                "Government": self._convert_value(df.loc[3, column].rstrip('%')),
                "Public": self._convert_value(df.loc[4, column].rstrip('%')),
                "Total_Shareholders": self._convert_value(df.loc[5, column].replace(',', ''))
            }
            self.data["shareholding_pattern"][column] = shareholding
            
    def process_cash_flows(self) -> None:
        """Process cash flow data"""
        cash_flow_section = self.soup.find('section', id='cash-flow')
        if not cash_flow_section:
            return
            
        table = cash_flow_section.find('table', class_='data-table')
        if not table:
            return
            
        headers = [th.text.strip() for th in table.find_all('th')]
        rows = []
        for tr in table.find_all('tr')[1:]:
            cells = [td.text.strip() for td in tr.find_all('td')]
            if cells:
                rows.append(cells)
                
        df = pd.DataFrame(rows, columns=headers)
        
        # Convert to dictionary format
        for column in df.columns[1:]:
            cash_flow = {
                "Operating_Cash_Flow": self._convert_value(df.loc[0, column]),
                "Investing_Cash_Flow": self._convert_value(df.loc[1, column]),
                "Financing_Cash_Flow": self._convert_value(df.loc[2, column]),
                "Net_Cash_Flow": self._convert_value(df.loc[3, column])
            }
            self.data["cash_flows"][column] = cash_flow
            
    @staticmethod
    def _convert_value(value: str) -> float:
        """Convert string values to float, handling special cases"""
        try:
            # Remove commas and convert to float
            cleaned_value = value.replace(',', '')
            return float(cleaned_value)
        except (ValueError, TypeError):
            return 0.0
            
    def analyze(self) -> None:
        """Perform complete analysis"""
        try:
            self.fetch_data()
            self.process_pros_cons()
            self.process_quarterly_data()
            self.process_financial_ratios()
            self.process_shareholding()
            self.process_cash_flows()
            print(f"Analysis completed for {self.symbol}")
        except Exception as e:
            print(f"Error during analysis: {e}")
            
    def get_latest_quarter_performance(self) -> Dict:
        """Get most recent quarter's performance"""
        if not self.data["quarterly_performance"]:
            return {}
        latest_quarter = max(self.data["quarterly_performance"].keys())
        return {
            "quarter": latest_quarter,
            "data": self.data["quarterly_performance"][latest_quarter]
        }
        
    def get_year_over_year_growth(self) -> Dict:
        """Calculate YoY growth rates"""
        quarters = list(self.data["quarterly_performance"].keys())
        if len(quarters) < 4:
            return {}
            
        current = self.data["quarterly_performance"][quarters[-1]]
        year_ago = self.data["quarterly_performance"][quarters[-4]]
        
        return {
            "Sales_Growth": ((current["Sales"] - year_ago["Sales"]) / year_ago["Sales"]) * 100,
            "Profit_Growth": ((current["Operating_Profit"] - year_ago["Operating_Profit"]) / 
                            year_ago["Operating_Profit"]) * 100,
            "OPM_Change": current["OPM"] - year_ago["OPM"]
        }
        
    def save_analysis(self, filename: Optional[str] = None) -> None:
        """Save analysis to JSON file"""
        if filename is None:
            filename = f"{self.symbol}_analysis.json"
            
        analysis = {
            "symbol": self.symbol,
            "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "data": self.data,
            "latest_quarter": self.get_latest_quarter_performance(),
            "yoy_growth": self.get_year_over_year_growth()
        }
        
        with open(filename, 'w') as f:
            json.dump(analysis, f, indent=4)
        print(f"Analysis saved to {filename}")
        
def analyze_stock(symbol: str) -> Dict:
    """Convenience function to analyze a stock"""
    analyzer = StockAnalyzer(symbol)
    analyzer.analyze()
    analyzer.save_analysis()
    return analyzer.data

# Example usage
if __name__ == "__main__":
    stock_data = analyze_stock("TATAMOTORS")
    
    # Print key metrics
    print("\nPros:")
    for pro in stock_data["pros"]:
        print(f"- {pro}")
        
    print("\nCons:")
    for con in stock_data["cons"]:
        print(f"- {con}")
        
    print("\nLatest Quarter Performance:")
    latest_quarter = max(stock_data["quarterly_performance"].keys())
    performance = stock_data["quarterly_performance"][latest_quarter]
    print(f"Quarter: {latest_quarter}")
    print(f"Sales: {performance['Sales']:,.2f}")
    print(f"Operating Profit: {performance['Operating_Profit']:,.2f}")
    print(f"OPM: {performance['OPM']}%")