# etl_server.py - ETL Flask Server
"""
Receives data from n8n workflows and processes through ETL pipeline.
Runs on port 5001.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from data.raw.transform import (
    transform_crypto_data,
    transform_stock_data,
    transform_holding_data,
    transform_registered_asset
)
from data.raw.load import (
    load_crypto_prices,
    load_stock_prices,
    load_holding,
    load_registered_asset,
    check_historical_data
)

app = Flask(__name__)
CORS(app)

# Logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'service': 'ETL Server',
        'port': 5001
    }), 200


@app.route('/api/etl/ingest', methods=['POST'])
def ingest_data():
    """
    Main ETL endpoint - receives data from n8n workflows.
    
    Expected JSON format:
    {
        "data_type": "crypto_prices" | "stock_prices" | "holding" | "registered_asset",
        "data": [...] or {...}
    }
    """
    try:
        payload = request.json
        print("recieved payload:", payload)
        
        if not payload:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
        
        data_type = payload.get('data_type')
        data = payload.get('data')
        
        if not data_type or data is None:
            return jsonify({'success': False, 'error': 'Missing data_type or data'}), 400
        
        logger.info(f"Received {data_type} data from n8n")
        
        # Route to appropriate ETL pipeline
        if data_type == 'crypto_prices':
            result = process_crypto_prices(data)
        elif data_type == 'stock_prices':
            result = process_stock_prices(data)
        elif data_type == 'holding':
            result = process_holding(data)
        elif data_type == 'registered_asset':
            result = process_registered_asset(data)
        else:
            return jsonify({
                'success': False,
                'error': f'Unknown data_type: {data_type}'
            }), 400
        
        if result['success']:
            logger.info(f"SUCCESS: {result['message']}")
            return jsonify(result), 200
        else:
            logger.error(f"ERROR: {result['message']}")
            return jsonify(result), 500
    
    except Exception as e:
        logger.error(f"ETL error: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'ETL processing failed: {str(e)}'
        }), 500


def process_crypto_prices(raw_data):
    """Extract → Transform → Load crypto prices."""
    try:
        # Transform
        df = transform_crypto_data(raw_data)
        logger.info(f"Transformed {len(df)} crypto price records")
        
        # Load
        result = load_crypto_prices(df)
        
        # Check if historical backfill is needed
        if result['success'] and len(df) > 0:
            for asset in df['asset'].unique():
                if not check_historical_data(asset, 'crypto', days_required=30):
                    logger.warning(f"WARNING: {asset} needs historical backfill")
                    result['needs_backfill'] = result.get('needs_backfill', [])
                    result['needs_backfill'].append(asset)
        
        return result
    
    except Exception as e:
        return {'success': False, 'message': f'Crypto ETL failed: {str(e)}'}


def process_stock_prices(raw_data):
    """Extract → Transform → Load stock prices."""
    try:
        # Accept both 'timestamp' and 'date' fields from n8n
        patched_data = []
        # Ensure raw_data is a list
        if isinstance(raw_data, dict):
            raw_data = [raw_data]
        for rec in raw_data:
            rec = dict(rec)  # copy
            if 'timestamp' not in rec and 'date' in rec:
                rec['timestamp'] = rec['date']
            patched_data.append(rec)
        # Transform
        df = transform_stock_data(patched_data)
        logger.info(f"Transformed {len(df)} stock price records")
        # Load
        result = load_stock_prices(df)
        # Check if historical backfill is needed
        if result['success'] and len(df) > 0:
            for symbol in df['symbol'].unique():
                if not check_historical_data(symbol, 'stock', days_required=30):
                    logger.warning(f"WARNING: {symbol} needs historical backfill")
                    result['needs_backfill'] = result.get('needs_backfill', [])
                    result['needs_backfill'].append(symbol)
        return result
    except Exception as e:
        return {'success': False, 'message': f'Stock ETL failed: {str(e)}'}


def process_holding(raw_data):
    """Extract → Transform → Load portfolio holding."""
    try:
        # Transform
        clean_data = transform_holding_data(raw_data)
        logger.info(f"Transformed holding: {clean_data['asset_symbol']}")
        
        # Load
        result = load_holding(clean_data)
        
        # Also register the asset if not already registered
        if result['success']:
            asset_result = load_registered_asset({
                'symbol': clean_data['asset_symbol'],
                'asset_type': clean_data['asset_type'],
                'name': clean_data.get('name', ''),
                'source': 'manual_entry'
            })
            result['asset_registered'] = asset_result['success']
            
            # Check if historical backfill is needed
            if not check_historical_data(clean_data['asset_symbol'], clean_data['asset_type'], days_required=30):
                logger.warning(f"WARNING: {clean_data['asset_symbol']} needs historical backfill")
                result['needs_backfill'] = True
                result['backfill_symbol'] = clean_data['asset_symbol']
                result['backfill_type'] = clean_data['asset_type']
        
        return result
    
    except Exception as e:
        return {'success': False, 'message': f'Holding ETL failed: {str(e)}'}


def process_registered_asset(raw_data):
    """Extract → Transform → Load registered asset."""
    try:
        # Transform
        clean_data = transform_registered_asset(raw_data)
        logger.info(f"Transformed registered asset: {clean_data['symbol']}")
        
        # Load
        result = load_registered_asset(clean_data)
        
        # Check if historical backfill is needed
        if result['success'] and not result.get('already_exists'):
            if not check_historical_data(clean_data['symbol'], clean_data['asset_type'], days_required=30):
                logger.warning(f"WARNING: {clean_data['symbol']} needs historical backfill")
                result['needs_backfill'] = True
                result['backfill_symbol'] = clean_data['symbol']
                result['backfill_type'] = clean_data['asset_type']
        
        return result
    
    except Exception as e:
        return {'success': False, 'message': f'Asset registration ETL failed: {str(e)}'}


@app.route('/api/etl/batch', methods=['POST'])
def batch_ingest():
    """
    Batch ETL endpoint for multiple data types.
    
    Expected JSON format:
    {
        "batches": [
            {"data_type": "crypto_prices", "data": [...]},
            {"data_type": "stock_prices", "data": [...]}
        ]
    }
    """
    try:
        payload = request.json
        batches = payload.get('batches', [])
        
        results = []
        for batch in batches:
            data_type = batch.get('data_type')
            data = batch.get('data')
            
            if data_type == 'crypto_prices':
                result = process_crypto_prices(data)
            elif data_type == 'stock_prices':
                result = process_stock_prices(data)
            elif data_type == 'holding':
                result = process_holding(data)
            elif data_type == 'registered_asset':
                result = process_registered_asset(data)
            else:
                result = {'success': False, 'message': f'Unknown data_type: {data_type}'}
            
            results.append({'data_type': data_type, 'result': result})
        
        return jsonify({
            'success': True,
            'batches_processed': len(results),
            'results': results
        }), 200
    
    except Exception as e:
        logger.error(f"Batch ETL error: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Batch ETL failed: {str(e)}'
        }), 500


if __name__ == '__main__':
    print("=" * 70)
    print("ETL Server Starting...")
    print("=" * 70)
    print(f"Database: {Path(project_root) / 'data' / 'database' / 'finance_data.db'}")
    print("\nEndpoints available:")
    print("   GET    /health")
    print("   POST   /api/etl/ingest")
    print("   POST   /api/etl/batch")
    print("\nExpected data_types:")
    print("   - crypto_prices")
    print("   - stock_prices")
    print("   - holding")
    print("   - registered_asset")
    print("\nServer running on http://localhost:5001")
    print("=" * 70)
    print()
    
    import subprocess
    import sys
    # Start Flask server
    app.run(host='0.0.0.0', port=5001, debug=False)
    # After server starts, run backfill_historical.py as a background process
    try:
        script_path = str(project_root / 'backfill_historical.py')
        print(f"Launching backfill_historical.py: {script_path}")
        subprocess.Popen([sys.executable, script_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        print(f"Failed to launch backfill_historical.py: {e}")
