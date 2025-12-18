
from typing import Dict, List, Any

def generate_single_asset_recommendation(asset_symbol: str, asset_type: str, current_price: float, quantity: float, total_cost: float) -> List[Dict[str, str]]:
    """
    Generate a rule-based recommendation for a single asset.
    """

    # ai_portfolio_advisor.py - AI-Powered Portfolio Recommendations
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        # Get price history if available
        if asset_type == 'crypto':
            cursor.execute('''SELECT price, timestamp FROM crypto_prices WHERE LOWER(asset) = LOWER(?) ORDER BY timestamp DESC LIMIT 30''', (asset_symbol,))
        else:
            cursor.execute('''SELECT close, timestamp FROM stock_prices WHERE UPPER(symbol) = UPPER(?) ORDER BY timestamp DESC LIMIT 30''', (asset_symbol,))
        rows = cursor.fetchall()
        prices = [row[0] for row in rows if row[0] is not None]
        conn.close()
    except Exception:
        prices = []
    gain_loss = (current_price * quantity) - total_cost
    gain_loss_pct = (gain_loss / total_cost) * 100 if total_cost > 0 else 0
    recs = []
    if gain_loss_pct > 50:
        recs.append({
            'type': 'success',
            'title': f'{asset_symbol} - Strong Gains',
            'message': f'Your position in {asset_symbol} is up {gain_loss_pct:.1f}%. Consider taking some profits.'
        })
    elif gain_loss_pct < -20:
        recs.append({
            'type': 'warning',
            'title': f'{asset_symbol} - Significant Loss',
            'message': f'Your position in {asset_symbol} is down {abs(gain_loss_pct):.1f}%. Review fundamentals and consider risk management.'
        })
    else:
        recs.append({
            'type': 'info',
            'title': f'{asset_symbol} - Stable Position',
            'message': f'Your position in {asset_symbol} is {gain_loss_pct:.1f}% from cost basis. Continue monitoring.'
        })
    # Add price trend if enough data
    if len(prices) >= 2:
        trend = prices[0] - prices[-1]
        if trend > 0:
            recs.append({'type': 'info', 'title': 'Recent Trend', 'message': f'{asset_symbol} has risen ${trend:.2f} over the last {len(prices)} periods.'})
        elif trend < 0:
            recs.append({'type': 'info', 'title': 'Recent Trend', 'message': f'{asset_symbol} has fallen ${abs(trend):.2f} over the last {len(prices)} periods.'})
    return recs
from typing import Dict, List, Any

# ai_portfolio_advisor.py - AI-Powered Portfolio Recommendations
"""
Generates AI-powered portfolio and single-asset recommendations using rule-based logic or AI APIs.
Can be used for:
 - Portfolio-level recommendations (default)
 - Single-asset recommendations (if asset details are provided)
Outputs JSON for easy integration.
"""

import sqlite3
from pathlib import Path
from typing import Dict, List, Any
import json
import argparse
import sys

# Database path
DB_PATH = Path(__file__).resolve().parent / "data" / "database" / "finance_data.db"


def get_db_connection():
    """Create database connection."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def get_portfolio_data() -> Dict[str, Any]:
    """
    Get comprehensive portfolio data for AI analysis.
    
    Returns:
        Dict with portfolio metrics, holdings, and performance data
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get all holdings with current prices
        cursor.execute('''
            SELECT asset_symbol, asset_type, quantity_owned, purchase_price, total_cost
            FROM portfolio_holdings
        ''')
        holdings_rows = cursor.fetchall()
        
        holdings = []
        total_invested = 0
        total_current_value = 0
        crypto_value = 0
        stock_value = 0
        
        for row in holdings_rows:
            asset_symbol = row['asset_symbol']
            asset_type = row['asset_type']
            quantity = row['quantity_owned']
            purchase_price = row['purchase_price']
            total_cost = row['total_cost'] or (quantity * purchase_price)
            
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
            profit_loss = current_value - total_cost
            profit_loss_pct = (profit_loss / total_cost) * 100 if total_cost > 0 else 0
            
            total_invested += total_cost
            total_current_value += current_value
            
            if asset_type == 'crypto':
                crypto_value += current_value
            else:
                stock_value += current_value
            
            holdings.append({
                'symbol': asset_symbol,
                'type': asset_type,
                'quantity': quantity,
                'purchase_price': purchase_price,
                'current_price': current_price,
                'total_cost': total_cost,
                'current_value': current_value,
                'profit_loss': profit_loss,
                'profit_loss_pct': profit_loss_pct
            })
        
        conn.close()
        
        # Calculate portfolio metrics
        total_pl = total_current_value - total_invested
        total_pl_pct = (total_pl / total_invested) * 100 if total_invested > 0 else 0
        crypto_allocation = (crypto_value / total_current_value) * 100 if total_current_value > 0 else 0
        stock_allocation = (stock_value / total_current_value) * 100 if total_current_value > 0 else 0
        
        return {
            'holdings': holdings,
            'total_invested': total_invested,
            'total_current_value': total_current_value,
            'total_profit_loss': total_pl,
            'total_pl_percentage': total_pl_pct,
            'crypto_value': crypto_value,
            'stock_value': stock_value,
            'crypto_allocation_pct': crypto_allocation,
            'stock_allocation_pct': stock_allocation,
            'holdings_count': len(holdings)
        }
    
    except Exception as e:
        print(f"Error getting portfolio data: {e}")
        return {}


def generate_rule_based_recommendations(portfolio_data: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Generate recommendations using rule-based AI logic.
    This is a fallback when no AI API is configured.
    """
    recommendations = []
    
    if not portfolio_data or not portfolio_data.get('holdings'):
        return [{
            'type': 'info',
            'title': 'Empty Portfolio',
            'message': 'Start by adding your first holding to receive personalized recommendations.'
        }]
    
    total_pl_pct = portfolio_data.get('total_pl_percentage', 0)
    crypto_alloc = portfolio_data.get('crypto_allocation_pct', 0)
    stock_alloc = portfolio_data.get('stock_allocation_pct', 0)
    total_value = portfolio_data.get('total_current_value', 0)
    holdings = portfolio_data.get('holdings', [])
    
    # Overall Performance Analysis
    if total_pl_pct > 20:
        recommendations.append({
            'type': 'success',
            'title': 'Strong Portfolio Performance',
            'message': f'Your portfolio is up {total_pl_pct:.2f}%, significantly outperforming typical market returns. Consider taking partial profits from overperforming positions to lock in gains and reduce risk exposure.'
        })
    elif total_pl_pct > 10:
        recommendations.append({
            'type': 'success',
            'title': 'Positive Portfolio Performance',
            'message': f'Your portfolio is up {total_pl_pct:.2f}%. This is solid growth. Continue monitoring positions and consider rebalancing if any single asset becomes too large a portion of your portfolio.'
        })
    elif total_pl_pct < -15:
        recommendations.append({
            'type': 'warning',
            'title': 'Portfolio Decline Alert',
            'message': f'Your portfolio is down {abs(total_pl_pct):.2f}%. Review underperforming positions and consider whether they still align with your investment thesis. This may be an opportunity to reassess your strategy.'
        })
    elif total_pl_pct < -5:
        recommendations.append({
            'type': 'info',
            'title': 'Minor Portfolio Decline',
            'message': f'Your portfolio is down {abs(total_pl_pct):.2f}%. Short-term losses are normal in volatile markets. Focus on your long-term strategy and avoid panic selling.'
        })
    
    # Diversification Analysis
    if crypto_alloc > 75:
        recommendations.append({
            'type': 'warning',
            'title': 'High Crypto Concentration Risk',
            'message': f'{crypto_alloc:.1f}% of your portfolio is in crypto. While crypto can offer high returns, this concentration exposes you to significant volatility. Consider adding stable stock positions to balance risk.'
        })
    elif crypto_alloc > 50:
        recommendations.append({
            'type': 'info',
            'title': 'Crypto-Heavy Allocation',
            'message': f'{crypto_alloc:.1f}% in crypto, {stock_alloc:.1f}% in stocks. Your portfolio leans heavily toward crypto. Consider if this aligns with your risk tolerance and investment timeline.'
        })
    elif crypto_alloc < 10 and crypto_alloc > 0:
        recommendations.append({
            'type': 'info',
            'title': 'Low Crypto Exposure',
            'message': f'Only {crypto_alloc:.1f}% of your portfolio is in crypto. If you\'re comfortable with higher risk, increasing crypto allocation could provide growth potential, though with increased volatility.'
        })
    elif stock_alloc > 90:
        recommendations.append({
            'type': 'info',
            'title': 'Stock-Focused Portfolio',
            'message': f'{stock_alloc:.1f}% of your portfolio is in stocks. This provides stability, but you might be missing growth opportunities in emerging asset classes like crypto or other alternatives.'
        })

    # Concentration Risk per Holding
    for holding in holdings:
        position_pct = (holding['current_value'] / total_value) * 100 if total_value > 0 else 0
        if position_pct > 40:
            recommendations.append({
                'type': 'error',
                'title': f'{holding["symbol"]} - Extreme Concentration',
                'message': f'{holding["symbol"]} represents {position_pct:.1f}% of your portfolio. This is dangerously concentrated. Consider selling portions to reduce risk, regardless of performance.'
            })
        elif position_pct > 25:
            recommendations.append({
                'type': 'warning',
                'title': f'{holding["symbol"]} - High Concentration',
                'message': f'{holding["symbol"]} is {position_pct:.1f}% of your portfolio. If this position moves against you, it could significantly impact overall returns. Consider rebalancing.'
            })
        # Individual Holding Performance
        if holding['profit_loss_pct'] > 100:
            recommendations.append({
                'type': 'success',
                'title': f'{holding["symbol"]} - Exceptional Gains',
                'message': f'{holding["symbol"]} is up {holding["profit_loss_pct"]:.1f}%! This is outstanding performance. Consider taking partial profits (sell 25-50%) to secure gains while maintaining upside exposure.'
            })
        elif holding['profit_loss_pct'] > 50:
            recommendations.append({
                'type': 'success',
                'title': f'{holding["symbol"]} - Strong Performance',
                'message': f'{holding["symbol"]} is up {holding["profit_loss_pct"]:.1f}%. Great return! Monitor this position and consider taking some profits if it continues climbing rapidly.'
            })
        elif holding['profit_loss_pct'] < -30:
            recommendations.append({
                'type': 'error',
                'title': f'{holding["symbol"]} - Significant Loss',
                'message': f'{holding["symbol"]} is down {abs(holding["profit_loss_pct"]):.1f}%. Evaluate whether the original investment thesis still holds. If fundamentals have deteriorated, cutting losses may be wise. If fundamentals are strong, this could be a buying opportunity.'
            })
        elif holding['profit_loss_pct'] < -15:
            pass
    
    # Portfolio Size Recommendations
    if len(holdings) == 1:
        recommendations.append({
            'type': 'warning',
            'title': 'Single Asset Portfolio',
            'message': 'Your entire portfolio is in one asset. This is extremely risky. Consider diversifying across multiple assets and asset types to reduce risk.'
        })
    elif len(holdings) == 2:
        recommendations.append({
            'type': 'info',
            'title': 'Limited Diversification',
            'message': 'You only hold 2 assets. Consider adding a few more positions (aim for 5-10) to achieve better diversification and reduce concentration risk.'
        })
    elif len(holdings) > 20:
        recommendations.append({
            'type': 'info',
            'title': 'High Number of Holdings',
            'message': f'You hold {len(holdings)} different assets. While diversification is good, too many holdings can dilute returns and make portfolio management difficult. Consider consolidating into your highest-conviction positions.'
        })
    
    return recommendations if recommendations else [{
        'type': 'success',
        'title': 'Well-Balanced Portfolio',
        'message': 'Your portfolio appears well-diversified with reasonable allocations. Continue monitoring your positions and stay disciplined with your investment strategy.'
    }]


def generate_ai_recommendations(portfolio_data: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Generate AI-powered recommendations.
    Currently uses rule-based logic. Can be extended to call OpenAI/Claude API.
    
    Args:
        portfolio_data: Dict with portfolio metrics and holdings
    
    Returns:
        List of recommendation dicts with type, title, message
    """
    # For now, use rule-based recommendations
    # TODO: Add OpenAI/Claude API integration here
    return generate_rule_based_recommendations(portfolio_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Portfolio/Asset Advisor")
    parser.add_argument('--asset_symbol', type=str, help='Asset symbol (for single asset)')
    parser.add_argument('--asset_type', type=str, help='Asset type (crypto or stock)')
    parser.add_argument('--current_price', type=float, help='Current price of asset')
    parser.add_argument('--quantity', type=float, help='Quantity held')
    parser.add_argument('--total_cost', type=float, help='Total cost basis')
    parser.add_argument('--json', action='store_true', help='Output as JSON')
    args = parser.parse_args()

    if args.asset_symbol and args.asset_type and args.current_price is not None and args.quantity is not None and args.total_cost is not None:
        # Single asset recommendation
        recs = generate_single_asset_recommendation(
            args.asset_symbol, args.asset_type, args.current_price, args.quantity, args.total_cost
        )
        if args.json:
            print(json.dumps(recs, indent=2))
        else:
            for i, rec in enumerate(recs, 1):
                print(f"{i}. [{rec['type'].upper()}] {rec['title']}")
                print(f"   {rec['message']}")
    else:
        # Portfolio-level recommendation
        portfolio = get_portfolio_data()
        if portfolio:
            recs = generate_ai_recommendations(portfolio)
            if args.json:
                print(json.dumps(recs, indent=2))
            else:
                print(f"\nPortfolio Summary:")
                print(f"  Total Value: ${portfolio['total_current_value']:,.2f}")
                print(f"  Total Invested: ${portfolio['total_invested']:,.2f}")
                print(f"  Profit/Loss: ${portfolio['total_profit_loss']:,.2f} ({portfolio['total_pl_percentage']:.2f}%)")
                print(f"  Holdings: {portfolio['holdings_count']}")
                print(f"  Crypto: {portfolio['crypto_allocation_pct']:.1f}% | Stocks: {portfolio['stock_allocation_pct']:.1f}%")
                print(f"\n{len(recs)} Recommendations Generated:")
                print("=" * 60)
                for i, rec in enumerate(recs, 1):
                    print(f"\n{i}. [{rec['type'].upper()}] {rec['title']}")
                    print(f"   {rec['message']}")
                print("\n" + "=" * 60)
        else:
            print(json.dumps([{'type': 'error', 'title': 'No Data', 'message': 'No portfolio data available.'}], indent=2) if args.json else "No portfolio data available.")
