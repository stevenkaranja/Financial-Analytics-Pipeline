# --- Imports and app initialization ---
from flask import Flask, request, jsonify
from flask_cors import CORS
import sqlite3
from datetime import datetime
import os

app = Flask(__name__)
CORS(app)

DB_PATH = r'C:\Users\HomePC\Documents\Automation Portfolio\financial-analytics-pipeline\data\database\finance_data.db'

# Store stock data from n8n workflow
@app.route('/api/store_stock_data', methods=['POST'])
def store_stock_data():
    """Store stock price data from n8n workflow"""
    try:
        data = request.json
        # Required fields: symbol, date, open, high, low, close, volume
        required = ['symbol', 'date', 'open', 'high', 'low', 'close', 'volume']
        if not all(field in data for field in required):
            return jsonify({'success': False, 'error': 'Missing required fields'}), 400

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO stock_prices (symbol, open, high, low, close, volume, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            data['symbol'],
            float(data['open']),
            float(data['high']),
            float(data['low']),
            float(data['close']),
            int(data['volume']),
            data['date']
        ))
        conn.commit()
        conn.close()
        return jsonify({'success': True}), 200
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

def get_db_connection():
    """Create database connection"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()}), 200

@app.route('/api/portfolio/holdings', methods=['GET'])
def get_portfolio_holdings():
    """Get all portfolio holdings"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, asset_symbol, asset_type, quantity_owned, purchase_price, 
                   purchase_date, total_cost, fees, notes, created_at, updated_at
            FROM portfolio_holdings
        ''')
        holdings = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return jsonify({'success': True, 'data': holdings}), 200
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# POST /api/portfolio/holdings endpoint REMOVED
# Holdings are now added via n8n workflows -> ETL server -> SQLite
# Streamlit triggers n8n webhook /webhook/asset_ingest instead of calling this API

@app.route('/api/portfolio/holdings/<int:holding_id>', methods=['PUT'])
def update_portfolio_holding(holding_id):
    """Update existing portfolio holding"""
    try:
        data = request.json
        conn = get_db_connection()
        cursor = conn.cursor()
        
        update_fields = []
        update_values = []
        
        allowed_fields = ['asset_symbol', 'asset_type', 'quantity_owned', 'purchase_price', 
                         'purchase_date', 'total_cost', 'fees', 'notes']
        
        for field in allowed_fields:
            if field in data:
                update_fields.append(f"{field} = ?")
                update_values.append(data[field])
        
        if not update_fields:
            return jsonify({'success': False, 'error': 'No fields to update'}), 400
        
        update_fields.append("updated_at = CURRENT_TIMESTAMP")
        update_values.append(holding_id)
        
        query = f"UPDATE portfolio_holdings SET {', '.join(update_fields)} WHERE id = ?"
        cursor.execute(query, update_values)
        conn.commit()
        
        if cursor.rowcount == 0:
            conn.close()
            return jsonify({'success': False, 'error': 'Holding not found'}), 404
        
        conn.close()
        return jsonify({'success': True}), 200
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/portfolio/holdings/<int:holding_id>', methods=['DELETE'])
def delete_portfolio_holding(holding_id):
    """Delete holding and all related asset data from all tables"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        # Get asset symbol and type for this holding
        cursor.execute('SELECT asset_symbol, asset_type FROM portfolio_holdings WHERE id = ?', (holding_id,))
        row = cursor.fetchone()
        if not row:
            conn.close()
            return jsonify({'success': False, 'error': 'Holding not found'}), 404
        asset_symbol, asset_type = row
        # Delete from portfolio_holdings
        cursor.execute('DELETE FROM portfolio_holdings WHERE id = ?', (holding_id,))
        # Delete from registered_assets
        cursor.execute('DELETE FROM registered_assets WHERE symbol = ? AND asset_type = ?', (asset_symbol, asset_type))
        # Delete from crypto_prices or stock_prices
        if asset_type == 'crypto':
            cursor.execute('DELETE FROM crypto_prices WHERE LOWER(asset) = LOWER(?)', (asset_symbol,))
        elif asset_type == 'stock':
            cursor.execute('DELETE FROM stock_prices WHERE UPPER(symbol) = UPPER(?)', (asset_symbol,))
        conn.commit()
        conn.close()
        return jsonify({'success': True}), 200
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/registered_assets', methods=['GET'])
def get_registered_assets():
    """Get all registered assets"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, symbol, asset_type, name, source, created_at
            FROM registered_assets
        ''')
        assets = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return jsonify({'success': True, 'data': assets}), 200
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# POST /api/registered_assets endpoint REMOVED
# Assets are now registered via n8n workflows -> ETL server -> SQLite
# ETL server auto-registers assets when processing holdings

@app.route('/api/registered_assets/<int:asset_id>', methods=['DELETE'])
def delete_registered_asset(asset_id):
    """Delete registered asset"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('DELETE FROM registered_assets WHERE id = ?', (asset_id,))
        conn.commit()
        
        if cursor.rowcount == 0:
            conn.close()
            return jsonify({'success': False, 'error': 'Asset not found'}), 404
        
        conn.close()
        return jsonify({'success': True}), 200
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/crypto_prices', methods=['GET'])
def get_crypto_prices():
    """Get crypto price history - row-based format"""
    try:
        asset = request.args.get('asset')
        limit = request.args.get('limit', 100, type=int)
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        if asset:
            cursor.execute('''
                SELECT asset, price, timestamp
                FROM crypto_prices
                WHERE LOWER(asset) = LOWER(?)
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (asset, limit))
        else:
            cursor.execute('''
                SELECT asset, price, timestamp
                FROM crypto_prices
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (limit,))
        
        prices = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return jsonify({'success': True, 'data': prices}), 200
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/stock_prices', methods=['GET'])
def get_stock_prices():
    """Get stock price history - row-based format"""
    try:
        symbol = request.args.get('symbol')
        limit = request.args.get('limit', 100, type=int)
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        if symbol:
            cursor.execute('''
                SELECT symbol, open, high, low, close, volume, timestamp
                FROM stock_prices
                WHERE UPPER(symbol) = UPPER(?)
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (symbol, limit))
        else:
            cursor.execute('''
                SELECT symbol, open, high, low, close, volume, timestamp
                FROM stock_prices
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (limit,))
        
        prices = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return jsonify({'success': True, 'data': prices}), 200
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/crypto/historical', methods=['GET'])
def get_crypto_historical():
    """Get historical crypto prices for dashboard charts"""
    try:
        days = request.args.get('days', 30, type=int)
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT asset, price, timestamp
            FROM crypto_prices
            WHERE timestamp >= datetime('now', '-' || ? || ' days')
            ORDER BY timestamp DESC
        ''', (days,))
        
        prices = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return jsonify({'success': True, 'data': prices}), 200
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/stock/historical', methods=['GET'])
def get_stock_historical():
    """Get historical stock prices for dashboard charts"""
    try:
        days = request.args.get('days', 30, type=int)
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT symbol, open, high, low, close, volume, timestamp
            FROM stock_prices
            WHERE timestamp >= datetime('now', '-' || ? || ' days')
            ORDER BY timestamp DESC
        ''', (days,))
        
        prices = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return jsonify({'success': True, 'data': prices}), 200
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/portfolio/summary', methods=['GET'])
def get_portfolio_summary():
    """Get portfolio summary with current prices"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get all holdings
        cursor.execute('''
            SELECT asset_symbol, asset_type, quantity_owned, purchase_price, total_cost
            FROM portfolio_holdings
        ''')
        holdings = cursor.fetchall()
        
        summary = {
            'total_value': 0,
            'total_cost': 0,
            'holdings': []
        }
        
        for holding in holdings:
            asset_symbol = holding[0]
            asset_type = holding[1]
            quantity = holding[2]
            purchase_price = holding[3]
            total_cost = holding[4] or (quantity * purchase_price)
            
            # Get latest price
            if asset_type == 'crypto':
                cursor.execute('''
                    SELECT price FROM crypto_prices 
                    WHERE LOWER(asset) = LOWER(?)
                    ORDER BY timestamp DESC LIMIT 1
                ''', (asset_symbol,))
            else:
                cursor.execute('''
                    SELECT close FROM stock_prices 
                    WHERE UPPER(symbol) = UPPER(?)
                    ORDER BY timestamp DESC LIMIT 1
                ''', (asset_symbol,))
            
            price_row = cursor.fetchone()
            current_price = price_row[0] if price_row else purchase_price
            current_value = quantity * current_price
            
            summary['total_value'] += current_value
            summary['total_cost'] += total_cost
            summary['holdings'].append({
                'asset_symbol': asset_symbol,
                'asset_type': asset_type,
                'quantity': quantity,
                'purchase_price': purchase_price,
                'current_price': current_price,
                'current_value': current_value,
                'profit_loss': current_value - total_cost
            })
        
        conn.close()
        return jsonify({'success': True, 'data': summary}), 200
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/portfolio/recommendations', methods=['GET'])
def get_portfolio_recommendations():
    """Get portfolio-level AI recommendations from ai_portfolio_advisor.py script"""
    try:
        # Import AI advisor functions
        import sys
        from pathlib import Path
        
        # Add parent directory to path to import ai_portfolio_advisor
        parent_dir = Path(__file__).resolve().parent.parent
        if str(parent_dir) not in sys.path:
            sys.path.insert(0, str(parent_dir))
        
        from ai_portfolio_advisor import get_portfolio_data, generate_ai_recommendations
        
        # Get portfolio data and generate recommendations
        portfolio_data = get_portfolio_data()
        
        if not portfolio_data:
            return jsonify({
                'success': True, 
                'data': [{
                    'type': 'info',
                    'title': 'No Portfolio Data',
                    'message': 'Add holdings to receive AI-powered portfolio recommendations.'
                }]
            }), 200
        
        recommendations = generate_ai_recommendations(portfolio_data)
        
        return jsonify({'success': True, 'data': recommendations}), 200
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    print("Flask API Server Starting...")
    print(f"Database: {DB_PATH}")
    print("\nRegistered Endpoints:")
    for rule in app.url_map.iter_rules():
        if rule.methods is not None:
            methods = ','.join(sorted(rule.methods - {'HEAD', 'OPTIONS'}))
        else:
            methods = ''
        print(f"  {methods:7} {rule}")
    print("\nServer running on http://localhost:5000\n")
    app.run(host='0.0.0.0', port=5000, debug=False)
