# scheduler/hourly_ingest.py - Hourly Data Ingestion Scheduler
"""
Queries SQLite for registered assets and triggers n8n hourly portfolio ingestion workflow.
Does NOT fetch prices directly - that's n8n's job.
"""

import sqlite3
import requests
from datetime import datetime
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

DB_PATH = project_root / "data" / "database" / "finance_data.db"
LOG_PATH = project_root / "data" / "logs" / "scheduler.log"

# n8n Configuration
N8N_BASE_URL = "https://n8n.datastagke.com"
N8N_HOURLY_WORKFLOW_URL = f"{N8N_BASE_URL}/webhook/hourly_portfolio_ingest"


def log(message: str):
    """Log message to file and console."""
    timestamp = datetime.now().isoformat()
    log_msg = f"[{timestamp}] {message}"
    print(log_msg)
    
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_PATH, 'a', encoding='utf-8') as f:
            f.write(log_msg + '\n')
    except:
        pass


def get_all_tracked_assets():
    """
    Get all assets to track (from registered_assets and portfolio_holdings).
    Returns list of tuples: [(symbol, asset_type), ...]
    """
    try:
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()
        
        # Get from registered_assets
        cursor.execute("""
            SELECT DISTINCT symbol, asset_type 
            FROM registered_assets
        """)
        registered = cursor.fetchall()
        
        # Get from portfolio_holdings
        cursor.execute("""
            SELECT DISTINCT asset_symbol, asset_type 
            FROM portfolio_holdings
        """)
        holdings = cursor.fetchall()
        
        conn.close()
        
        # Combine and deduplicate
        all_assets = {}
        for symbol, asset_type in registered + holdings:
            key = f"{symbol.upper()}_{asset_type.lower()}"
            all_assets[key] = (symbol.upper(), asset_type.lower())
        
        return list(all_assets.values())
    
    except Exception as e:
        log(f"ERROR: Failed to get tracked assets: {e}")
        return []


def trigger_n8n_hourly_workflow(assets):
    """
    Send asset list to n8n hourly portfolio ingestion workflow.
    
    Args:
        assets: List of tuples [(symbol, asset_type), ...]
    
    Returns:
        Dict with success status and message
    """
    if not assets:
        return {"success": False, "message": "No assets to process"}
    
    # Format assets for n8n
    crypto_assets = [symbol for symbol, atype in assets if atype == 'crypto']
    stock_assets = [symbol for symbol, atype in assets if atype == 'stock']
    
    payload = {
        "crypto_assets": crypto_assets,
        "stock_assets": stock_assets,
        "timestamp": datetime.now().isoformat(),
        "total_count": len(assets)
    }
    
    try:
        log(f"Triggering n8n hourly workflow with {len(crypto_assets)} crypto + {len(stock_assets)} stock assets")
        
        response = requests.post(
            N8N_HOURLY_WORKFLOW_URL,
            json=payload,
            timeout=30,
            headers={'Content-Type': 'application/json'}
        )
        
        if response.status_code in [200, 201]:
            result = response.json() if response.text else {}
            log(f"SUCCESS: n8n workflow triggered - {result.get('message', 'OK')}")
            return {
                "success": True,
                "message": f"Triggered n8n for {len(assets)} assets",
                "n8n_response": result
            }
        else:
            log(f"WARNING: n8n returned status {response.status_code}: {response.text[:200]}")
            return {
                "success": False,
                "message": f"n8n workflow failed with status {response.status_code}"
            }
    
    except requests.exceptions.Timeout:
        log("ERROR: n8n workflow timeout (30s)")
        return {"success": False, "message": "n8n workflow timeout"}
    
    except requests.exceptions.ConnectionError:
        log("ERROR: Cannot connect to n8n. Is n8n running and accessible?")
        return {"success": False, "message": "Cannot connect to n8n"}
    
    except Exception as e:
        log(f"ERROR: Failed to trigger n8n workflow: {e}")
        return {"success": False, "message": str(e)}


def run_hourly_ingestion():
    """Main function - get assets and trigger n8n workflow."""
    log("=" * 60)
    log("Starting hourly data ingestion scheduler")
    
    # Get all tracked assets
    assets = get_all_tracked_assets()
    
    if not assets:
        log("WARNING: No assets to track")
        log("=" * 60)
        return
    
    log(f"Found {len(assets)} assets to track")
    
    # Trigger n8n workflow
    result = trigger_n8n_hourly_workflow(assets)
    
    if result['success']:
        log(f"SUCCESS: {result['message']}")
    else:
        log(f"FAILED: {result['message']}")
    
    log("=" * 60)


if __name__ == "__main__":
    try:
        run_hourly_ingestion()
    except KeyboardInterrupt:
        log("Scheduler interrupted by user")
    except Exception as e:
        log(f"FATAL ERROR: {e}")
        sys.exit(1)
