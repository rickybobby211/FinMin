import finnhub, os, dotenv
from datetime import datetime
dotenv.load_dotenv('../.env')
dotenv.load_dotenv('.env')
key = os.getenv('FINNHUB_API_KEY')
if not key:
    print("No key")
else:
    client = finnhub.Client(api_key=key)
    res = client.general_news('general', min_id=0)
    print(f"Total: {len(res)}")
    if res:
        print(f"First: {datetime.fromtimestamp(res[0]['datetime'])}")
        print(f"Last: {datetime.fromtimestamp(res[-1]['datetime'])}")
