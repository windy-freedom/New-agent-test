from flask import Flask, render_template, jsonify
import random
from datetime import datetime, timedelta
import threading
import time
import json
from collections import deque

app = Flask(__name__)

# 模拟加密货币数据
crypto_data = {
    'BTC': {'name': '比特币', 'price': 45000, 'change': 2.5},
    'ETH': {'name': '以太坊', 'price': 3000, 'change': 1.8},
    'BNB': {'name': '币安币', 'price': 380, 'change': -0.5}
}

# 资产组合状态
portfolio = {
    'BTC': {'amount': 0.5, 'avg_price': 44000},
    'ETH': {'amount': 5.0, 'avg_price': 2900},
    'BNB': {'amount': 10.0, 'avg_price': 375},
    'USDT': 100000  # 初始USDT余额
}

# 模拟交易历史
trading_history = []

# AI交易状态
ai_trading_active = False
ai_threads = {}

# AI学习记录
learning_history = []

# AI代理配置
AI_AGENTS = {
    'AI-保守型': {
        'risk_factor': 0.3,
        'min_change_threshold': 1.5,
        'learning_rate': 0.1,
        'performance': {'profit': 0, 'trades': 0, 'success_rate': 0},
        'strategy_weights': {'price_change': 0.4, 'volume': 0.3, 'trend': 0.3},
        'learned_from': set(),
        'teaching_count': 0
    },
    'AI-均衡型': {
        'risk_factor': 0.5,
        'min_change_threshold': 1.0,
        'learning_rate': 0.2,
        'performance': {'profit': 0, 'trades': 0, 'success_rate': 0},
        'strategy_weights': {'price_change': 0.33, 'volume': 0.33, 'trend': 0.34},
        'learned_from': set(),
        'teaching_count': 0
    },
    'AI-激进型': {
        'risk_factor': 0.7,
        'min_change_threshold': 0.8,
        'learning_rate': 0.3,
        'performance': {'profit': 0, 'trades': 0, 'success_rate': 0},
        'strategy_weights': {'price_change': 0.5, 'volume': 0.2, 'trend': 0.3},
        'learned_from': set(),
        'teaching_count': 0
    }
}

# 价格历史记录
price_history = {crypto: deque(maxlen=10) for crypto in crypto_data.keys()}

def calculate_total_assets():
    """计算总资产价值（以USDT计价）"""
    total = portfolio['USDT']
    for crypto, data in portfolio.items():
        if crypto != 'USDT':
            total += data['amount'] * crypto_data[crypto]['price']
    return round(total, 2)

def calculate_pnl(crypto):
    """计算某个加密货币的收益"""
    if crypto not in portfolio or crypto == 'USDT':
        return 0
    current_value = portfolio[crypto]['amount'] * crypto_data[crypto]['price']
    cost_value = portfolio[crypto]['amount'] * portfolio[crypto]['avg_price']
    return round(current_value - cost_value, 2)

def update_portfolio(crypto, action, amount, price):
    """更新资产组合"""
    total_cost = amount * price
    
    if action == '买入':
        if portfolio['USDT'] >= total_cost:
            portfolio['USDT'] -= total_cost
            if crypto not in portfolio:
                portfolio[crypto] = {'amount': 0, 'avg_price': 0}
            # 更新平均价格
            old_value = portfolio[crypto]['amount'] * portfolio[crypto]['avg_price']
            new_value = amount * price
            total_amount = portfolio[crypto]['amount'] + amount
            portfolio[crypto]['avg_price'] = (old_value + new_value) / total_amount
            portfolio[crypto]['amount'] = total_amount
            return True
        return False
    else:  # 卖出
        if crypto in portfolio and portfolio[crypto]['amount'] >= amount:
            portfolio['USDT'] += total_cost
            portfolio[crypto]['amount'] -= amount
            if portfolio[crypto]['amount'] == 0:
                portfolio[crypto]['avg_price'] = 0
            return True
        return False

def calculate_trend(crypto):
    """计算价格趋势"""
    if len(price_history[crypto]) < 2:
        return 0
    prices = list(price_history[crypto])
    return sum(b - a for a, b in zip(prices[:-1], prices[1:])) / (len(prices) - 1)

def update_strategy_weights(agent_name, success):
    """更新AI策略权重"""
    agent = AI_AGENTS[agent_name]
    learning_rate = agent['learning_rate']
    
    if success:
        for key in agent['strategy_weights']:
            current_weight = agent['strategy_weights'][key]
            agent['strategy_weights'][key] = current_weight + learning_rate * (1 - current_weight)
    else:
        for key in agent['strategy_weights']:
            current_weight = agent['strategy_weights'][key]
            agent['strategy_weights'][key] = current_weight - learning_rate * current_weight

    total_weight = sum(agent['strategy_weights'].values())
    for key in agent['strategy_weights']:
        agent['strategy_weights'][key] /= total_weight

def share_knowledge(source_agent, target_agent):
    """AI代理之间共享知识"""
    if AI_AGENTS[source_agent]['performance']['success_rate'] > AI_AGENTS[target_agent]['performance']['success_rate']:
        # 记录学习历史
        learning_event = {
            'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'teacher': source_agent,
            'student': target_agent,
            'old_weights': dict(AI_AGENTS[target_agent]['strategy_weights']),
            'teacher_success_rate': AI_AGENTS[source_agent]['performance']['success_rate']
        }
        
        # 更新策略权重
        for key in AI_AGENTS[target_agent]['strategy_weights']:
            source_weight = AI_AGENTS[source_agent]['strategy_weights'][key]
            target_weight = AI_AGENTS[target_agent]['strategy_weights'][key]
            AI_AGENTS[target_agent]['strategy_weights'][key] = (source_weight + target_weight) / 2
        
        # 更新学习记录
        learning_event['new_weights'] = dict(AI_AGENTS[target_agent]['strategy_weights'])
        learning_history.append(learning_event)
        
        # 更新学习关系
        AI_AGENTS[target_agent]['learned_from'].add(source_agent)
        AI_AGENTS[source_agent]['teaching_count'] += 1

def ai_trading_strategy(agent_name, crypto):
    """改进的AI交易策略"""
    agent = AI_AGENTS[agent_name]
    current_price = crypto_data[crypto]['price']
    current_change = crypto_data[crypto]['change']
    trend = calculate_trend(crypto)
    
    weights = agent['strategy_weights']
    score = (
        weights['price_change'] * (abs(current_change) / agent['min_change_threshold']) +
        weights['volume'] * random.random() +
        weights['trend'] * (trend / current_price)
    )
    
    threshold = agent['risk_factor']
    
    if score > threshold:
        return 'buy' if current_change < 0 or trend > 0 else 'sell'
    return None

def ai_trading_loop(agent_name):
    """AI交易主循环"""
    while ai_trading_active:
        for crypto in crypto_data.keys():
            if not ai_trading_active:
                break
                
            price_history[crypto].append(crypto_data[crypto]['price'])
            
            action = ai_trading_strategy(agent_name, crypto)
            if action:
                amount = random.uniform(0.1, 0.5)
                price = crypto_data[crypto]['price']
                total = amount * price
                
                # 执行交易并更新资产组合
                trade_action = '买入' if action == 'buy' else '卖出'
                success = update_portfolio(crypto, trade_action, amount, price)
                
                if success:
                    AI_AGENTS[agent_name]['performance']['trades'] += 1
                    AI_AGENTS[agent_name]['performance']['profit'] += calculate_pnl(crypto)
                    
                    trades = AI_AGENTS[agent_name]['performance']['trades']
                    AI_AGENTS[agent_name]['performance']['success_rate'] = (
                        AI_AGENTS[agent_name]['performance']['profit'] / (trades * total) if trades > 0 else 0
                    )
                    
                    update_strategy_weights(agent_name, success)
                    
                    # 与其他AI共享知识
                    for other_agent in AI_AGENTS:
                        if other_agent != agent_name:
                            share_knowledge(agent_name, other_agent)
                    
                    trading_history.append({
                        'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'crypto': crypto,
                        'action': trade_action,
                        'amount': round(amount, 4),
                        'price': round(price, 2),
                        'total': round(total, 2),
                        'type': f'AI交易 ({agent_name})',
                        'success': success
                    })
            
            time.sleep(2)
        time.sleep(5)

@app.route('/')
def index():
    return render_template('index.html', crypto_data=crypto_data, ai_agents=AI_AGENTS)

@app.route('/api/update_prices')
def update_prices():
    for crypto in crypto_data.values():
        change = random.uniform(-2, 2)
        crypto['price'] *= (1 + change/100)
        crypto['change'] = change
    return jsonify(crypto_data)

@app.route('/api/portfolio')
def get_portfolio():
    """获取资产组合状态"""
    total_assets = calculate_total_assets()
    pnl_data = {crypto: calculate_pnl(crypto) for crypto in crypto_data.keys()}
    
    return jsonify({
        'portfolio': portfolio,
        'total_assets': total_assets,
        'pnl': pnl_data
    })

@app.route('/api/trade/<crypto>/<action>')
def trade(crypto, action):
    if crypto not in crypto_data:
        return jsonify({'error': '无效的加密货币'}), 400
    
    amount = random.uniform(0.1, 1.0)
    price = crypto_data[crypto]['price']
    total = amount * price
    
    trade_action = '买入' if action == 'buy' else '卖出'
    success = update_portfolio(crypto, trade_action, amount, price)
    
    if success:
        trading_history.append({
            'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'crypto': crypto,
            'action': trade_action,
            'amount': round(amount, 4),
            'price': round(price, 2),
            'total': round(total, 2),
            'type': '手动交易'
        })
        return jsonify({'success': True, 'trade': trading_history[-1]})
    else:
        return jsonify({'error': '余额不足或持仓不足'}), 400

@app.route('/api/history')
def get_history():
    return jsonify(trading_history)

@app.route('/api/learning_history')
def get_learning_history():
    """获取AI学习历史"""
    return jsonify(learning_history)

@app.route('/api/ai_trading/<status>')
def toggle_ai_trading(status):
    global ai_trading_active
    
    if status == 'start' and not ai_trading_active:
        ai_trading_active = True
        for agent_name in AI_AGENTS:
            ai_threads[agent_name] = threading.Thread(
                target=ai_trading_loop,
                args=(agent_name,)
            )
            ai_threads[agent_name].start()
        return jsonify({'status': 'started'})
    elif status == 'stop' and ai_trading_active:
        ai_trading_active = False
        for thread in ai_threads.values():
            thread.join(timeout=1)
        ai_threads.clear()
        return jsonify({'status': 'stopped'})
    
    return jsonify({'status': 'unchanged'})

@app.route('/api/ai_trading/status')
def get_ai_trading_status():
    return jsonify({
        'active': ai_trading_active,
        'agents': {
            name: {
                'performance': agent['performance'],
                'strategy_weights': agent['strategy_weights'],
                'learned_from': list(agent['learned_from']),
                'teaching_count': agent['teaching_count']
            }
            for name, agent in AI_AGENTS.items()
        }
    })

if __name__ == '__main__':
    app.run(debug=True, threaded=True, port=5001)