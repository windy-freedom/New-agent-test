from flask import Flask, render_template, jsonify
import random
from datetime import datetime, timedelta
import threading
import time
import json
from collections import deque
import os
from dotenv import load_dotenv
import openai

# 加载环境变量
load_dotenv()

app = Flask(__name__)

# 配置OpenAI
openai.api_key = os.getenv('OPENAI_API_KEY')

# 大模型分析历史
llm_analysis_history = []

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
    """整合大模型的AI交易策略"""
    agent = AI_AGENTS[agent_name]
    current_price = crypto_data[crypto]['price']
    current_change = crypto_data[crypto]['change']
    trend = calculate_trend(crypto)
    
    # 获取大模型分析
    analysis = get_market_analysis(crypto)
    llm_action, suggested_position = extract_trading_signal(analysis)
    
    # 结合传统策略和大模型建议
    weights = agent['strategy_weights']
    traditional_score = (
        weights['price_change'] * (abs(current_change) / agent['min_change_threshold']) +
        weights['volume'] * random.random() +
        weights['trend'] * (trend / current_price)
    )
    
    threshold = agent['risk_factor']
    
    # 如果大模型给出明确建议，增加其权重
    if llm_action:
        if traditional_score > threshold:
            # 当传统策略和大模型建议一致时，增加确信度
            traditional_action = 'buy' if current_change < 0 or trend > 0 else 'sell'
            if llm_action == traditional_action:
                return llm_action, suggested_position
        # 大模型建议的权重随着AI代理的成功率增加
        if agent['performance']['success_rate'] > 0.5:
            return llm_action, suggested_position
            
    # 当大模型无明确建议时，依靠传统策略
    if traditional_score > threshold:
        action = 'buy' if current_change < 0 or trend > 0 else 'sell'
        return action, 0.1
        
    return None, 0

def ai_trading_loop(agent_name):
    """AI交易主循环"""
    while ai_trading_active:
        for crypto in crypto_data.keys():
            if not ai_trading_active:
                break
                
            price_history[crypto].append(crypto_data[crypto]['price'])
            
            action, position = ai_trading_strategy(agent_name, crypto)
            if action:
                # 根据建议仓位计算交易数量
                available_funds = portfolio['USDT'] if action == 'buy' else \
                    portfolio[crypto]['amount'] * crypto_data[crypto]['price']
                amount = (available_funds * position) / crypto_data[crypto]['price']
                amount = min(amount, random.uniform(0.1, 1.0))  # 设置上限以控制风险
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

def get_market_analysis(crypto):
    """使用大模型分析市场状况"""
    try:
        # 准备市场数据
        current_price = crypto_data[crypto]['price']
        current_change = crypto_data[crypto]['change']
        trend = calculate_trend(crypto)
        price_history_list = list(price_history[crypto])
        
        try:
            # 尝试调用OpenAI API
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "你是一个专业的加密货币交易分析师，擅长技术分析和风险管理。"},
                    {"role": "user", "content": f"""
作为加密货币交易分析师，请分析以下{crypto}市场数据并给出建议：
- 当前价格: {current_price}
- 24小时涨跌幅: {current_change}%
- 价格趋势: {trend}
- 近期价格历史: {price_history_list}

请从以下几个方面进行分析：
1. 市场趋势判断
2. 潜在风险分析
3. 交易建议（买入/卖出/观望）
4. 建议的仓位大小（占总资金的百分比）

请用中文回答，简明扼要。
"""}
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            analysis = response.choices[0].message.content
            
            # 记录分析历史
            llm_analysis_history.append({
                'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'crypto': crypto,
                'analysis': analysis,
                'market_data': {
                    'price': current_price,
                    'change': current_change,
                    'trend': trend
                }
            })
            
            return analysis
        except Exception as api_error:
            print(f"OpenAI API调用失败: {str(api_error)}")
            return f"分析失败: {str(api_error)}"
            
    except Exception as e:
        return f"分析出错: {str(e)}"

def extract_trading_signal(analysis):
    """从大模型分析结果中提取交易信号"""
    analysis = analysis.lower()
    
    # 提取交易建议
    if '买入' in analysis or 'buy' in analysis:
        action = 'buy'
    elif '卖出' in analysis or 'sell' in analysis:
        action = 'sell'
    else:
        action = None
        
    # 提取建议仓位
    import re
    position_matches = re.findall(r'(\d+)%', analysis)
    suggested_position = float(position_matches[0])/100 if position_matches else 0.1
    
    return action, suggested_position

@app.route('/api/market_analysis/<crypto>')
def get_crypto_analysis(crypto):
    """获取特定加密货币的市场分析"""
    if crypto not in crypto_data:
        return jsonify({'error': '无效的加密货币'}), 400
        
    analysis = get_market_analysis(crypto)
    return jsonify({
        'crypto': crypto,
        'analysis': analysis,
        'current_price': crypto_data[crypto]['price'],
        'current_change': crypto_data[crypto]['change']
    })

@app.route('/api/analysis_history')
def get_analysis_history():
    """获取分析历史"""
    return jsonify(llm_analysis_history)

if __name__ == '__main__':
    app.run(debug=True, threaded=True, port=5002)