"""
Script för att extrahera den predikterade procenten från backtest CSV-filer
och lägga till den som en ny kolumn.
"""
import pandas as pd
import re
import sys

def extract_predicted_percentage(text):
    """
    Extraherar den predikterade procenten från modellens svar.
    Hanterar olika format:
    - "DOWN by 2%"
    - "DOWN by 2-4%"
    - "Direction: DOWN" och "Percentage Change: -3% to -5%"
    - "Prediction: DOWN by 5%"
    """
    if not text or not isinstance(text, str):
        return None
    
    text_lower = text.lower()
    
    # Pattern 1: "**Prediction:** **DOWN** by **2%**" (med bold markers)
    pattern1 = r'\*\*prediction[:\s\*\*]+\*\*\s*\*\*(up|down)\*\*\s+by\s+\*\*(\d+(?:\.\d+)?)(?:\s*-\s*(\d+(?:\.\d+)?))?\s*%\*\*'
    match = re.search(pattern1, text_lower, re.IGNORECASE)
    if match:
        direction = -1 if match.group(1).lower() == 'down' else 1
        pct1 = float(match.group(2))
        pct2 = float(match.group(3)) if match.group(3) else None
        if pct2:
            avg_pct = (pct1 + pct2) / 2
            return avg_pct * direction
        return pct1 * direction
    
    # Pattern 2: "Prediction: DOWN by 2%" eller "Prediction: DOWN by 2-4%"
    pattern2 = r'prediction[:\s\*\*]+\s*(?:direction[:\s\*\*]+)?\s*(up|down)\s+by\s+(\d+(?:\.\d+)?)(?:\s*-\s*(\d+(?:\.\d+)?))?\s*%'
    match = re.search(pattern2, text_lower, re.IGNORECASE)
    if match:
        direction = -1 if match.group(1).lower() == 'down' else 1
        pct1 = float(match.group(2))
        pct2 = float(match.group(3)) if match.group(3) else None
        if pct2:
            avg_pct = (pct1 + pct2) / 2
            return avg_pct * direction
        return pct1 * direction
    
    # Pattern 3: "Direction: DOWN" och "Percentage Change: -3% to -5%"
    direction_match = re.search(r'direction[:\s\*\*]+\s*(up|down)', text_lower, re.IGNORECASE)
    pct_match = re.search(r'percentage\s+change[:\s\*\*]+\s*(-?\d+(?:\.\d+)?)(?:\s*(?:to|-)\s*(-?\d+(?:\.\d+)?))?\s*%', text_lower, re.IGNORECASE)
    
    if direction_match and pct_match:
        direction = -1 if direction_match.group(1).lower() == 'down' else 1
        pct1 = float(pct_match.group(1))
        pct2 = float(pct_match.group(2)) if pct_match.group(2) else None
        
        # Om procenten redan har tecken, använd den direkt
        if pct1 < 0 or (pct_match.group(1).startswith('-')):
            if pct2:
                avg_pct = (abs(pct1) + abs(pct2)) / 2
                return -avg_pct if pct1 < 0 else avg_pct
            return pct1
        else:
            if pct2:
                avg_pct = (pct1 + pct2) / 2
            else:
                avg_pct = pct1
            return avg_pct * direction
    
    # Pattern 4: "DOWN by 2%" (enklare format, utan "Prediction:")
    pattern4 = r'(?:^|\n|\.)\s*(up|down)\s+by\s+(\d+(?:\.\d+)?)(?:\s*-\s*(\d+(?:\.\d+)?))?\s*%'
    match = re.search(pattern4, text_lower, re.IGNORECASE | re.MULTILINE)
    if match:
        direction = -1 if match.group(1).lower() == 'down' else 1
        pct1 = float(match.group(2))
        pct2 = float(match.group(3)) if match.group(3) else None
        if pct2:
            avg_pct = (pct1 + pct2) / 2
            return avg_pct * direction
        return pct1 * direction
    
    return None

def process_backtest_file(input_file, output_file=None):
    """
    Läser en backtest CSV-fil, extraherar predikterade procenten och lägger till en ny kolumn.
    """
    print(f"Läser fil: {input_file}")
    df = pd.read_csv(input_file)
    
    if 'full_text' not in df.columns:
        print("Fel: Filen saknar 'full_text' kolumnen")
        return
    
    print(f"Bearbetar {len(df)} rader...")
    
    # Extrahera predikterad procent
    df['predicted_pct'] = df['full_text'].apply(extract_predicted_percentage)
    
    # Räkna hur många som lyckades extraheras
    extracted_count = df['predicted_pct'].notna().sum()
    print(f"Extraherade predikterad procent för {extracted_count}/{len(df)} rader ({extracted_count/len(df)*100:.1f}%)")
    
    # Visa några exempel där det inte gick att extrahera
    failed = df[df['predicted_pct'].isna()]
    if len(failed) > 0:
        print(f"\nVarning: Kunde inte extrahera predikterad procent för {len(failed)} rader")
        print("Första raden där det misslyckades:")
        if len(failed) > 0:
            first_failed = failed.iloc[0]
            print(f"  Date: {first_failed['date']}")
            print(f"  Prediction: {first_failed['prediction']}")
            # Visa relevant del av texten
            text = str(first_failed['full_text'])
            if 'prediction' in text.lower():
                idx = text.lower().find('prediction')
                print(f"  Relevant text: {text[max(0, idx-50):idx+200]}")
    
    # Spara resultatet
    if output_file is None:
        output_file = input_file.replace('.csv', '_with_predicted_pct.csv')
    
    df.to_csv(output_file, index=False)
    print(f"\nResultat sparad till: {output_file}")
    
    # Visa sammanfattning
    print("\n=== Sammanfattning ===")
    print(f"Totala rader: {len(df)}")
    print(f"Lyckade extraktioner: {extracted_count}")
    print(f"Misslyckade: {len(df) - extracted_count}")
    
    if extracted_count > 0:
        print(f"\nStatistik för predikterad procent:")
        print(f"  Medelvärde: {df['predicted_pct'].mean():.2f}%")
        print(f"  Median: {df['predicted_pct'].median():.2f}%")
        print(f"  Min: {df['predicted_pct'].min():.2f}%")
        print(f"  Max: {df['predicted_pct'].max():.2f}%")
        
        # Jämför med faktisk procent
        if 'pct_change' in df.columns:
            print(f"\nJämförelse med faktisk procent (pct_change):")
            comparison = df[df['predicted_pct'].notna() & df['pct_change'].notna()].copy()
            if len(comparison) > 0:
                comparison.loc[:, 'diff'] = abs(comparison['predicted_pct'] - comparison['pct_change'])
                print(f"  Genomsnittligt absolut fel: {comparison['diff'].mean():.2f}%")
                print(f"  Median absolut fel: {comparison['diff'].median():.2f}%")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Användning: python extract_predicted_pct.py <input_file.csv> [output_file.csv]")
        print("\nExempel:")
        print("  python extract_predicted_pct.py backtest_AAPL_2025-01-03_5w.csv")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    process_backtest_file(input_file, output_file)

