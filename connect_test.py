import os
#!/usr/bin/env python3
"""Connection test for TopstepX — authenticate + list tradable accounts.
Never prints the API key."""
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from broker import TopstepXClient
from config import TOPSTEPX_USERNAME, TOPSTEPX_API_KEY

if not TOPSTEPX_USERNAME or not TOPSTEPX_API_KEY:
    print("FAIL: TOPSTEPX_USERNAME/TOPSTEPX_API_KEY missing (check .env)")
    sys.exit(1)

print(f"username: {TOPSTEPX_USERNAME}")
print(f"api key: {'set (' + str(len(TOPSTEPX_API_KEY)) + ' chars)' if TOPSTEPX_API_KEY else 'MISSING'}")

b = TopstepXClient(TOPSTEPX_USERNAME, TOPSTEPX_API_KEY)
try:
    b.authenticate()
    print("AUTH OK — token acquired")
except Exception as e:
    print(f"AUTH FAILED: {e}")
    sys.exit(1)

try:
    acct = b.pick_account()
    print(f"ACCOUNT OK — selected: {acct['name']} (id={acct['id']}, balance=${acct.get('balance', '?')})")
except Exception as e:
    print(f"ACCOUNT FAILED: {e}")
    sys.exit(1)
