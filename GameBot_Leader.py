import streamlit as st
import json
from datetime import datetime
import google.generativeai as genai
import plotly.graph_objects as go
import pandas as pd
import sqlite3
from pathlib import Path

# Configure Gemini API
genai.configure(api_key=st.secrets.get("GEMINI_API_KEY", ""))

# Page config
st.set_page_config(page_title="SCM Game Bot", page_icon="🏭", layout="wide")

# Custom CSS for better visuals
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.1rem;
        margin-top: 0;
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    .metric-box {
        background: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Database setup
DB_PATH = Path("scm_game_leaderboard.db")

def init_database():
    """Initialize SQLite database for leaderboard"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS leaderboard (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            player_name TEXT NOT NULL,
            client_type TEXT NOT NULL,
            avg_score REAL NOT NULL,
            cost_efficiency INTEGER NOT NULL,
            customer_satisfaction INTEGER NOT NULL,
            resilience INTEGER NOT NULL,
            sustainability INTEGER NOT NULL,
            timestamp TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Create index for faster queries
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_avg_score 
        ON leaderboard(avg_score DESC)
    """)
    
    conn.commit()
    conn.close()

def add_to_database(player_name, client_type, scores):
    """Add player's performance to database"""
    avg_score = sum(scores.values()) / len(scores)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO leaderboard 
        (player_name, client_type, avg_score, cost_efficiency, 
         customer_satisfaction, resilience, sustainability, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        player_name,
        client_type,
        round(avg_score, 1),
        scores['cost_efficiency'],
        scores['customer_satisfaction'],
        scores['resilience'],
        scores['sustainability'],
        timestamp
    ))
    
    conn.commit()
    conn.close()
    
    return cursor.lastrowid

def get_leaderboard(limit=10):
    """Get top scores from database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT player_name, client_type, avg_score, 
               cost_efficiency, customer_satisfaction, 
               resilience, sustainability, timestamp
        FROM leaderboard
        ORDER BY avg_score DESC, created_at DESC
        LIMIT ?
    """, (limit,))
    
    results = cursor.fetchall()
    conn.close()
    
    leaderboard = []
    for row in results:
        leaderboard.append({
            'name': row[0],
            'client': row[1],
            'avg_score': row[2],
            'cost_efficiency': row[3],
            'customer_satisfaction': row[4],
            'resilience': row[5],
            'sustainability': row[6],
            'timestamp': row[7]
        })
    
    return leaderboard

def get_player_rank(player_name, current_avg_score):
    """Get player's rank in the leaderboard"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Get count of scores better than current
    cursor.execute("""
        SELECT COUNT(*) + 1
        FROM leaderboard
        WHERE avg_score > ?
    """, (current_avg_score,))
    
    rank = cursor.fetchone()[0]
    conn.close()
    
    return rank

def get_total_players():
    """Get total number of players in database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(DISTINCT player_name) FROM leaderboard")
    count = cursor.fetchone()[0]
    
    conn.close()
    return count

def get_player_stats(player_name):
    """Get statistics for a specific player"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT COUNT(*), AVG(avg_score), MAX(avg_score), MIN(avg_score)
        FROM leaderboard
        WHERE player_name = ?
    """, (player_name,))
    
    result = cursor.fetchone()
    conn.close()
    
    if result[0] > 0:
        return {
            'games_played': result[0],
            'avg_score': round(result[1], 1),
            'best_score': round(result[2], 1),
            'worst_score': round(result[3], 1)
        }
    return None

# Initialize database on app start
init_database()

# Initialize session state
if 'game_state' not in st.session_state:
    st.session_state.game_state = {
        'stage': 0,
        'scenario': 0,
        'client_type': 'TechCo',
        'scores': {
            'cost_efficiency': 100,
            'customer_satisfaction': 100,
            'resilience': 100,
            'sustainability': 100
        },
        'decisions': [],
        'feedback_history': [],
        'current_scenario': None,
        'decision_made': False,
        'selected_choice': None,
        'player_name': None
    }

# SCM Game Prompt Template
SCM_GAME_PROMPT = """You are an AI Game Master for a Supply Chain Management training simulation. 

GAME CONTEXT:
- Player role: Supply Chain Consultant
- Client: {client_type}
- Current Stage: {stage_name}
- Scenario: {scenario_number}
- Current Scores: Cost Efficiency: {cost}%, Customer Satisfaction: {satisfaction}%, Resilience: {resilience}%, Sustainability: {sustainability}%

GAME STAGES:
1. Planning (Demand forecasting, inventory strategy)
2. Sourcing (Supplier selection, procurement)
3. Manufacturing (Production planning, quality control)
4. Delivery/Logistics (Transportation, distribution)
5. Returns/After-sales (Defect management, recycling)

YOUR TASK:
Generate a realistic supply chain scenario for {client_type} at the {stage_name} stage.

IMPORTANT: The impacts should be significant. Use values between -15 and +15 for each metric.
Make trade-offs meaningful - good decisions should have both positive and negative impacts.

FORMAT YOUR RESPONSE AS VALID JSON ONLY (no markdown, no code blocks):
{{
    "scenario_title": "Brief title",
    "scenario_description": "Detailed description of the challenge (2-3 sentences)",
    "context": "Additional business context if needed",
    "options": [
        {{
            "id": "A",
            "text": "Option description",
            "impact": {{
                "cost_efficiency": -15 to +15,
                "customer_satisfaction": -15 to +15,
                "resilience": -15 to +15,
                "sustainability": -15 to +15
            }},
            "feedback": "What happens if this option is chosen"
        }},
        {{
            "id": "B",
            "text": "Option description",
            "impact": {{
                "cost_efficiency": -15 to +15,
                "customer_satisfaction": -15 to +15,
                "resilience": -15 to +15,
                "sustainability": -15 to +15
            }},
            "feedback": "What happens if this option is chosen"
        }},
        {{
            "id": "C",
            "text": "Option description",
            "impact": {{
                "cost_efficiency": -15 to +15,
                "customer_satisfaction": -15 to +15,
                "resilience": -15 to +15,
                "sustainability": -15 to +15
            }},
            "feedback": "What happens if this option is chosen"
        }},
        {{
            "id": "D",
            "text": "Option description",
            "impact": {{
                "cost_efficiency": -15 to +15,
                "customer_satisfaction": -15 to +15,
                "resilience": -15 to +15,
                "sustainability": -15 to +15
            }},
            "feedback": "What happens if this option is chosen"
        }}
    ],
    "learning_point": "Key SCM concept illustrated by this scenario"
}}

Previous decisions: {previous_decisions}
Make the scenario logically connected to previous choices when applicable.
Return ONLY the JSON object, no additional text or markdown formatting.
"""

# Stage definitions
STAGES = [
    "Planning",
    "Sourcing", 
    "Manufacturing",
    "Delivery/Logistics",
    "Returns/After-sales"
]

# Client options
CLIENT_TYPES = {
    "TechCo": "A smartphone manufacturer facing global supply chain challenges",
    "FMCG Corp": "A fast-moving consumer goods company with extensive distribution needs",
    "PharmaCare": "A pharmaceutical manufacturer with strict quality requirements",
    "AutoDrive": "An automotive manufacturer dealing with complex supplier networks"
}


def extract_json_from_text(text):
    """Extract JSON from text that might contain markdown code blocks or extra text"""
    # Try to find JSON in code blocks
    if "```json" in text:
        start = text.find("```json") + 7
        end = text.find("```", start)
        text = text[start:end].strip()
    elif "```" in text:
        start = text.find("```") + 3
        end = text.find("```", start)
        text = text[start:end].strip()
    
    # Try to find JSON object boundaries
    start_idx = text.find("{")
    end_idx = text.rfind("}") + 1
    
    if start_idx != -1 and end_idx > start_idx:
        text = text[start_idx:end_idx]
    
    return text


def get_scenario_from_gemini(client_type, stage_name, scenario_number, scores, previous_decisions):
    """Fetch scenario from Gemini API with robust error handling"""
    
    # Format the prompt
    prompt = SCM_GAME_PROMPT.format(
        client_type=client_type,
        stage_name=stage_name,
        scenario_number=scenario_number,
        cost=scores['cost_efficiency'],
        satisfaction=scores['customer_satisfaction'],
        resilience=scores['resilience'],
        sustainability=scores['sustainability'],
        previous_decisions=json.dumps(previous_decisions[-3:]) if previous_decisions else "None"
    )
    
    try:
        # Create model instance
        model = genai.GenerativeModel("gemini-2.5-flash-lite")
        
        # Call the model with JSON mode
        response = model.generate_content(
            prompt,
            generation_config={
                "response_mime_type": "application/json",
                "temperature": 0.7
            }
        )
        
        # Get response text
        response_text = response.text
        
        # Try to parse JSON
        try:
            scenario = json.loads(response_text)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown or extra text
            cleaned_text = extract_json_from_text(response_text)
            scenario = json.loads(cleaned_text)
        
        # Validate scenario structure
        required_fields = ['scenario_title', 'scenario_description', 'options', 'learning_point']
        if not all(field in scenario for field in required_fields):
            raise ValueError("Missing required fields in scenario")
        
        # Validate options
        if len(scenario['options']) < 3:
            raise ValueError("Not enough options in scenario")
        
        for option in scenario['options']:
            if not all(key in option for key in ['id', 'text', 'impact', 'feedback']):
                raise ValueError("Invalid option structure")
        
        return scenario
        
    except Exception as e:
        st.error(f"⚠️ Error generating scenario: {str(e)}")
        # Return a fallback scenario
        return create_fallback_scenario(stage_name)


def create_fallback_scenario(stage_name):
    """Create a fallback scenario if LLM fails"""
    
    fallback_scenarios = {
        "Planning": {
            "scenario_title": "Demand Forecasting Crisis",
            "scenario_description": "Market analysts predict a 40% surge in demand next quarter, but your current inventory capacity can only handle 20% growth. You need to decide how to prepare your supply chain.",
            "context": "Historical data shows similar surges have led to stockouts and lost customers.",
            "options": [
                {
                    "id": "A",
                    "text": "Increase inventory levels aggressively by 50% to meet projected demand",
                    "impact": {"cost_efficiency": -12, "customer_satisfaction": 10, "resilience": 8, "sustainability": -8},
                    "feedback": "You captured market opportunity but increased holding costs and waste. Excess inventory ties up capital."
                },
                {
                    "id": "B",
                    "text": "Use just-in-time ordering with multiple backup suppliers",
                    "impact": {"cost_efficiency": 5, "customer_satisfaction": -5, "resilience": 12, "sustainability": 8},
                    "feedback": "Lower inventory costs but some delivery delays occurred. Strong supplier network helped manage risks."
                },
                {
                    "id": "C",
                    "text": "Maintain current levels and implement demand management strategies",
                    "impact": {"cost_efficiency": 8, "customer_satisfaction": -10, "resilience": -5, "sustainability": 5},
                    "feedback": "Controlled costs but lost market share to competitors who stocked adequately."
                },
                {
                    "id": "D",
                    "text": "Partner with third-party logistics for flexible capacity scaling",
                    "impact": {"cost_efficiency": -5, "customer_satisfaction": 8, "resilience": 10, "sustainability": 3},
                    "feedback": "Moderate costs with good flexibility. Partnership fees reduced margins slightly."
                }
            ],
            "learning_point": "Demand forecasting requires balancing inventory costs against potential lost sales. Flexibility often outperforms pure efficiency."
        },
        "Sourcing": {
            "scenario_title": "Supplier Reliability vs Cost",
            "scenario_description": "Your primary supplier offers the lowest prices but has a history of quality issues and late deliveries. Alternative suppliers cost 15% more but have better track records.",
            "context": "Recent customer complaints about product defects have increased by 25%.",
            "options": [
                {
                    "id": "A",
                    "text": "Switch entirely to premium suppliers despite higher costs",
                    "impact": {"cost_efficiency": -15, "customer_satisfaction": 12, "resilience": 10, "sustainability": 8},
                    "feedback": "Quality improved dramatically and customer satisfaction soared, but profit margins tightened significantly."
                },
                {
                    "id": "B",
                    "text": "Maintain current supplier but implement strict quality controls",
                    "impact": {"cost_efficiency": -8, "customer_satisfaction": 5, "resilience": -5, "sustainability": 0},
                    "feedback": "Quality improved somewhat but inspection costs added up. Some defects still reached customers."
                },
                {
                    "id": "C",
                    "text": "Use a dual-sourcing strategy with 60% from current, 40% from premium",
                    "impact": {"cost_efficiency": -8, "customer_satisfaction": 8, "resilience": 12, "sustainability": 5},
                    "feedback": "Balanced approach provided flexibility and improved overall quality while managing costs."
                },
                {
                    "id": "D",
                    "text": "Invest in supplier development program for current vendor",
                    "impact": {"cost_efficiency": -10, "customer_satisfaction": 3, "resilience": 5, "sustainability": 10},
                    "feedback": "Long-term investment showed promise but immediate improvements were limited. Building strong partnerships takes time."
                }
            ],
            "learning_point": "Sourcing decisions must balance cost, quality, and risk. The cheapest option often has hidden costs in quality issues and disruptions."
        },
        "Manufacturing": {
            "scenario_title": "Production Line Breakdown",
            "scenario_description": "A critical manufacturing line has failed, reducing capacity by 30%. Repairs will take 2 weeks. You have orders to fulfill and penalties for late delivery.",
            "context": "Peak season is approaching and competitors are actively courting your customers.",
            "options": [
                {
                    "id": "A",
                    "text": "Outsource production to a contract manufacturer at premium rates",
                    "impact": {"cost_efficiency": -15, "customer_satisfaction": 10, "resilience": 5, "sustainability": -5},
                    "feedback": "Met all delivery deadlines but significantly reduced profit margins. Customer loyalty remained strong."
                },
                {
                    "id": "B",
                    "text": "Run overtime shifts on remaining lines at 150% capacity",
                    "impact": {"cost_efficiency": -10, "customer_satisfaction": 5, "resilience": -8, "sustainability": -12},
                    "feedback": "Partially met demand but equipment stress led to additional failures. Worker fatigue increased defect rates."
                },
                {
                    "id": "C",
                    "text": "Prioritize key customers and negotiate delayed delivery for others",
                    "impact": {"cost_efficiency": 5, "customer_satisfaction": -12, "resilience": 3, "sustainability": 5},
                    "feedback": "Protected important relationships but lost several medium-sized accounts to competitors."
                },
                {
                    "id": "D",
                    "text": "Deploy emergency repair team with expedited parts at high cost",
                    "impact": {"cost_efficiency": -12, "customer_satisfaction": 8, "resilience": 10, "sustainability": 0},
                    "feedback": "Line restored in 5 days instead of 14. High repair costs but avoided most customer penalties."
                }
            ],
            "learning_point": "Manufacturing disruptions require quick decisions. Having contingency plans and backup capacity is crucial for resilience."
        },
        "Delivery/Logistics": {
            "scenario_title": "Transportation Cost Surge",
            "scenario_description": "Fuel prices have spiked 40% and shipping costs have doubled due to global events. Your logistics budget is now severely strained.",
            "context": "Customers expect free or low-cost shipping, and competitors are absorbing some costs.",
            "options": [
                {
                    "id": "A",
                    "text": "Pass shipping costs to customers through price increases",
                    "impact": {"cost_efficiency": 10, "customer_satisfaction": -15, "resilience": 5, "sustainability": 0},
                    "feedback": "Protected margins but lost price-sensitive customers. Sales volume dropped 20%."
                },
                {
                    "id": "B",
                    "text": "Absorb all increased costs to maintain competitive pricing",
                    "impact": {"cost_efficiency": -15, "customer_satisfaction": 10, "resilience": -5, "sustainability": -5},
                    "feedback": "Maintained customer base but quarterly profits turned negative. Unsustainable long-term."
                },
                {
                    "id": "C",
                    "text": "Optimize routes and consolidate shipments to reduce frequency",
                    "impact": {"cost_efficiency": 8, "customer_satisfaction": -5, "resilience": 8, "sustainability": 12},
                    "feedback": "Reduced costs through efficiency but slightly longer delivery times. Environmental benefits were positive."
                },
                {
                    "id": "D",
                    "text": "Switch to multimodal transport and negotiate long-term carrier contracts",
                    "impact": {"cost_efficiency": 5, "customer_satisfaction": 3, "resilience": 12, "sustainability": 8},
                    "feedback": "Secured stable rates and improved flexibility. Diversified transport options reduced vulnerability."
                }
            ],
            "learning_point": "Logistics optimization balances cost, speed, and reliability. Building strong carrier relationships and route flexibility is key."
        },
        "Returns/After-sales": {
            "scenario_title": "Product Recall Decision",
            "scenario_description": "Quality testing has revealed a potential defect in 5% of your shipped products. The defect causes minor performance issues but no safety risks.",
            "context": "News of the issue has leaked on social media. Your reputation is at stake.",
            "options": [
                {
                    "id": "A",
                    "text": "Issue full recall and replace all potentially affected units",
                    "impact": {"cost_efficiency": -15, "customer_satisfaction": 15, "resilience": 8, "sustainability": -10},
                    "feedback": "Customers praised your proactive response. High costs but brand loyalty increased significantly."
                },
                {
                    "id": "B",
                    "text": "Offer free repairs only to customers who report issues",
                    "impact": {"cost_efficiency": 5, "customer_satisfaction": -8, "resilience": -5, "sustainability": 3},
                    "feedback": "Lower costs but negative PR escalated. Social media sentiment turned against your brand."
                },
                {
                    "id": "C",
                    "text": "Implement enhanced warranty and preventive service program",
                    "impact": {"cost_efficiency": -8, "customer_satisfaction": 10, "resilience": 10, "sustainability": 5},
                    "feedback": "Balanced approach that caught defects early. Built trust through extended support."
                },
                {
                    "id": "D",
                    "text": "Provide software updates and extended monitoring for all units",
                    "impact": {"cost_efficiency": -5, "customer_satisfaction": 5, "resilience": 8, "sustainability": 12},
                    "feedback": "Innovative solution that prevented waste. Some customers wanted physical replacements though."
                }
            ],
            "learning_point": "Returns management affects brand reputation and customer lifetime value. Proactive response to quality issues builds trust despite short-term costs."
        }
    }
    
    return fallback_scenarios.get(stage_name, {
        "scenario_title": f"{stage_name} Challenge",
        "scenario_description": "Your team needs to make a critical decision to optimize the supply chain.",
        "context": "This is a standard scenario while we resolve technical issues.",
        "options": [
            {
                "id": "A",
                "text": "Take the conservative approach focusing on cost savings",
                "impact": {"cost_efficiency": 10, "customer_satisfaction": -5, "resilience": 0, "sustainability": -5},
                "feedback": "You saved costs but may have compromised other areas."
            },
            {
                "id": "B",
                "text": "Balance all factors with a moderate investment",
                "impact": {"cost_efficiency": 0, "customer_satisfaction": 5, "resilience": 5, "sustainability": 5},
                "feedback": "A balanced approach that maintains stability."
            },
            {
                "id": "C",
                "text": "Invest heavily in innovation and sustainability",
                "impact": {"cost_efficiency": -10, "customer_satisfaction": 10, "resilience": 10, "sustainability": 15},
                "feedback": "Higher costs but strong long-term benefits."
            }
        ],
        "learning_point": f"Understanding trade-offs in {stage_name} decisions"
    })


def generate_performance_analysis(scores, decisions, feedback_history, client_type):
    """Generate detailed performance analysis using Gemini"""
    
    # Calculate metrics
    score_changes = {metric: value - 100 for metric, value in scores.items()}
    avg_score = sum(scores.values()) / len(scores)
    
    # Build decision summary
    decision_summary = []
    for decision in decisions:
        decision_summary.append(f"Stage: {decision['stage']}, Choice: {decision['choice']}")
    
    prompt = f"""You are an expert Supply Chain Management consultant providing personalized feedback to a trainee.

CLIENT: {client_type}
FINAL SCORES:
- Cost Efficiency: {scores['cost_efficiency']}% (Change: {score_changes['cost_efficiency']:+d}%)
- Customer Satisfaction: {scores['customer_satisfaction']}% (Change: {score_changes['customer_satisfaction']:+d}%)
- Resilience: {scores['resilience']}% (Change: {score_changes['resilience']:+d}%)
- Sustainability: {scores['sustainability']}% (Change: {score_changes['sustainability']:+d}%)
- Average Score: {avg_score:.1f}%

DECISIONS MADE:
{chr(10).join(decision_summary)}

Provide a detailed, personalized analysis in JSON format:
{{
    "overview": "2-3 sentence summary of their overall performance and decision-making pattern",
    "strengths": ["3-4 specific strengths based on their scores and decisions"],
    "improvements": ["3-4 specific areas where they can improve, with actionable advice"],
    "personal_learnings": ["4-5 key learnings they should take away from this game based on their actual decisions"],
    "recommendations": "A paragraph of specific recommendations for how they can apply these learnings in real-world SCM scenarios"
}}

Make the analysis highly personalized based on their actual scores and decision patterns. Be constructive and specific.
Return ONLY the JSON object."""

    try:
        model = genai.GenerativeModel("gemini-2.5-flash-lite")
        response = model.generate_content(
            prompt,
            generation_config={
                "response_mime_type": "application/json",
                "temperature": 0.7
            }
        )
        
        response_text = response.text
        
        try:
            analysis = json.loads(response_text)
        except json.JSONDecodeError:
            cleaned_text = extract_json_from_text(response_text)
            analysis = json.loads(cleaned_text)
        
        return analysis
        
    except Exception as e:
        st.error(f"Could not generate detailed analysis: {str(e)}")
        return {
            "overview": "You completed the simulation successfully!",
            "strengths": ["Completed all stages", "Made strategic decisions"],
            "improvements": ["Consider balancing all metrics", "Think about long-term impacts"],
            "personal_learnings": ["Supply chains require trade-offs", "Different decisions affect different metrics"],
            "recommendations": "Continue learning about supply chain management principles."
        }


def calculate_score_change(current_scores, impact):
    """Calculate new scores based on impact"""
    new_scores = {}
    for metric, value in current_scores.items():
        if metric in impact:
            new_value = value + impact[metric]
            new_scores[metric] = max(0, min(100, new_value))
        else:
            new_scores[metric] = value
    return new_scores


def render_dashboard(scores):
    """Create performance dashboard"""
    fig = go.Figure()
    
    categories = ['Cost Efficiency', 'Customer Satisfaction', 'Resilience', 'Sustainability']
    values = [scores['cost_efficiency'], scores['customer_satisfaction'], 
              scores['resilience'], scores['sustainability']]
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Performance',
        line_color='rgb(102, 126, 234)',
        fillcolor='rgba(102, 126, 234, 0.5)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickfont=dict(size=10)
            )
        ),
        showlegend=False,
        height=400,
        margin=dict(l=80, r=80, t=40, b=40)
    )
    
    return fig


def render_progress_bar():
    """Render game progress"""
    stage = st.session_state.game_state['stage']
    scenario = st.session_state.game_state['scenario']
    
    if stage > 0 and stage <= len(STAGES):
        total_scenarios = len(STAGES) * 2  # 2 scenarios per stage
        current_progress = ((stage - 1) * 2 + scenario)
        progress = current_progress / total_scenarios
        
        st.markdown("### 📊 Game Progress")
        st.progress(progress)
        st.caption(f"Stage {stage} of {len(STAGES)} • Scenario {scenario + 1} of 2")


def add_to_leaderboard(player_name, client_type, scores):
    """Add player's performance to leaderboard (legacy function for compatibility)"""
    return add_to_database(player_name, client_type, scores)


def render_leaderboard():
    """Display the leaderboard from database"""
    leaderboard = get_leaderboard(10)
    total_players = get_total_players()
    
    if not leaderboard:
        st.info("🏆 Be the first to complete the game and claim the top spot!")
        return
    
    st.markdown(f"### 🏆 Top 10 Supply Chain Consultants")
    st.caption(f"Total players: {total_players}")
    
    # Create leaderboard dataframe
    df = pd.DataFrame(leaderboard)
    
    # Add rank column
    df.insert(0, 'Rank', range(1, len(df) + 1))
    
    # Format for display
    display_df = df[['Rank', 'name', 'client', 'avg_score', 'cost_efficiency', 
                     'customer_satisfaction', 'resilience', 'sustainability', 'timestamp']]
    
    display_df.columns = ['🏅 Rank', '👤 Player', '🏢 Client', '⭐ Avg Score', 
                          '💰 Cost', '😊 Customer', '🛡️ Resilience', '🌱 Sustain', '📅 Date']
    
    # Style the dataframe with custom colors
    def style_leaderboard(row):
        # Highlight current player
        if row['👤 Player'] == st.session_state.game_state.get('player_name'):
            return ['background-color: #fffacd; color: #000000; font-weight: bold'] * len(row)
        # Color top 3 differently
        elif row['🏅 Rank'] == 1:
            return ['background-color: #FFD700; color: #000000; font-weight: bold'] * len(row)
        elif row['🏅 Rank'] == 2:
            return ['background-color: #C0C0C0; color: #000000; font-weight: bold'] * len(row)
        elif row['🏅 Rank'] == 3:
            return ['background-color: #CD7F32; color: #000000; font-weight: bold'] * len(row)
        return ['color: #1f1f1f'] * len(row)
    
    styled_df = display_df.style.apply(style_leaderboard, axis=1)
    
    st.dataframe(styled_df, width='stretch', hide_index=True)
    
    # Add medals for top 3
    if len(leaderboard) >= 1:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if len(leaderboard) >= 1:
                leader = leaderboard[0]
                st.markdown(f"""
                🥇 **1st Place**  
                **{leader['name']}**  
                Score: {leader['avg_score']}%  
                Client: {leader['client']}
                """)
        
        with col2:
            if len(leaderboard) >= 2:
                second = leaderboard[1]
                st.markdown(f"""
                🥈 **2nd Place**  
                **{second['name']}**  
                Score: {second['avg_score']}%  
                Client: {second['client']}
                """)
        
        with col3:
            if len(leaderboard) >= 3:
                third = leaderboard[2]
                st.markdown(f"""
                🥉 **3rd Place**  
                **{third['name']}**  
                Score: {third['avg_score']}%  
                Client: {third['client']}
                """)


def main():
    # Header
    st.markdown('<h1 class="main-header">🏭 Supply Chain Management Game</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Step into the role of a Supply Chain Consultant and navigate real-world challenges</p>', unsafe_allow_html=True)
    
    # Sidebar for game info
    with st.sidebar:
        st.markdown("## 📊 Game Status")
        
        if st.session_state.game_state['stage'] > 0:
            # Current stage display
            stage_idx = st.session_state.game_state['stage'] - 1
            if stage_idx < len(STAGES):
                st.info(f"**Current Stage:** {STAGES[stage_idx]}")
            
            # Progress bar
            render_progress_bar()
            
            st.markdown("---")
            st.markdown("### Performance Metrics")
            
            # Score cards with colors
            scores = st.session_state.game_state['scores']
            
            for metric, value in scores.items():
                label = metric.replace('_', ' ').title()
                if value >= 80:
                    color = "🟢"
                elif value >= 60:
                    color = "🟡"
                else:
                    color = "🔴"
                
                st.metric(
                    f"{color} {label}", 
                    f"{value}%",
                    delta=f"{value - 100:+d}%"
                )
        
        st.markdown("---")
        
        if st.button("🔄 Restart Game", type="secondary", use_container_width=True):
            # Keep player name for next game
            player_name = st.session_state.game_state.get('player_name')
            st.session_state.game_state = {
                'stage': 0,
                'scenario': 0,
                'client_type': 'TechCo',
                'scores': {
                    'cost_efficiency': 100,
                    'customer_satisfaction': 100,
                    'resilience': 100,
                    'sustainability': 100
                },
                'decisions': [],
                'feedback_history': [],
                'current_scenario': None,
                'decision_made': False,
                'selected_choice': None,
                'player_name': player_name
            }
            st.rerun()
    
    # Main game area
    if st.session_state.game_state['stage'] == 0:
        # Introduction screen
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown("### Welcome to the SCM Simulation! 🎮")
            
            # Show leaderboard first at the top
            st.markdown("---")
            render_leaderboard()
            st.markdown("---")
            
            # Player name input
            if not st.session_state.game_state.get('player_name'):
                st.markdown("### 👤 Enter Your Name")
                player_name = st.text_input("Your name will appear on the leaderboard:", 
                                            placeholder="Enter your name",
                                            label_visibility="collapsed")
                
                if player_name and len(player_name.strip()) > 0:
                    st.session_state.game_state['player_name'] = player_name.strip()
                    st.success(f"Welcome, {player_name}! 👋")
                    
                    # Show player stats if they've played before
                    stats = get_player_stats(player_name.strip())
                    if stats:
                        st.info(f"""
                        **Your Stats:**  
                        🎮 Games Played: {stats['games_played']}  
                        ⭐ Average Score: {stats['avg_score']}%  
                        🏆 Best Score: {stats['best_score']}%
                        """)
                else:
                    st.warning("Please enter your name to continue")
            else:
                player_name = st.session_state.game_state['player_name']
                st.success(f"Welcome back, {player_name}! 👋")
                
                # Show player stats
                stats = get_player_stats(player_name)
                if stats:
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("🎮 Games Played", stats['games_played'])
                    with col_b:
                        st.metric("⭐ Avg Score", f"{stats['avg_score']}%")
                    with col_c:
                        st.metric("🏆 Best Score", f"{stats['best_score']}%")
            
            st.markdown("---")
            
            with st.expander("📚 What is Supply Chain Management?", expanded=True):
                st.markdown("""
                Supply Chain Management ensures products move smoothly from planning to delivery. It covers **5 critical stages**:
                
                1. **📋 Planning** - Demand forecasting & inventory strategy
                2. **🤝 Sourcing** - Supplier selection & procurement
                3. **🏭 Manufacturing** - Production planning & quality control
                4. **🚚 Delivery** - Transportation & distribution logistics
                5. **↩️ Returns** - Defect management & recycling
                """)
            
            with st.expander("🎯 Understanding Key Metrics"):
                st.markdown("""
                Your decisions will be evaluated across four critical dimensions:
                
                **💰 Cost Efficiency**
                - Measures how well you manage operational costs and resource utilization
                - Includes procurement costs, inventory holding costs, and operational expenses
                - Higher scores indicate better financial performance
                
                **😊 Customer Satisfaction**
                - Reflects how well you meet customer expectations
                - Includes delivery speed, product quality, and service reliability
                - Critical for brand reputation and repeat business
                
                **🛡️ Resilience**
                - Measures your supply chain's ability to handle disruptions
                - Includes backup suppliers, contingency planning, and risk management
                - Essential for business continuity during crises
                
                **🌱 Sustainability**
                - Evaluates environmental and social responsibility
                - Includes carbon footprint, ethical sourcing, and waste management
                - Increasingly important for modern businesses and regulations
                
                *Note: These metrics often compete with each other. Great supply chain managers find the right balance!*
                """)
            
            with st.expander("🎯 Game Rules"):
                st.markdown("""
                - Play as a **Supply Chain Consultant**
                - Solve **real-world challenges** for your client
                - Navigate through **5 stages**, each with multiple scenarios
                - Your choices impact **4 key metrics** (sometimes positively, sometimes negatively)
                - Receive instant feedback and learning points
                - Get a detailed performance analysis at the end
                """)
            
            st.markdown("---")
            st.markdown("### 🏢 Select Your Client")
            
            client = st.selectbox(
                "Choose your client company:",
                options=list(CLIENT_TYPES.keys()),
                format_func=lambda x: f"{x} - {CLIENT_TYPES[x]}",
                label_visibility="collapsed"
            )
            st.session_state.game_state['client_type'] = client
            
            st.markdown("")
            
            if st.button("🎮 Start Game", type="primary", width='stretch'):
                if st.session_state.game_state.get('player_name'):
                    st.session_state.game_state['stage'] = 1
                    st.session_state.game_state['scenario'] = 0
                    st.rerun()
                else:
                    st.error("Please enter your name first!")
    
    elif st.session_state.game_state['stage'] <= len(STAGES):
        # Game in progress
        stage_idx = st.session_state.game_state['stage'] - 1
        current_stage = STAGES[stage_idx]
        
        st.markdown(f"## 🎯 Stage {st.session_state.game_state['stage']}: {current_stage}")
        
        # Get or use cached scenario
        if not st.session_state.game_state['decision_made'] and st.session_state.game_state['current_scenario'] is None:
            with st.spinner("🤔 Generating scenario..."):
                scenario = get_scenario_from_gemini(
                    st.session_state.game_state['client_type'],
                    current_stage,
                    st.session_state.game_state['scenario'],
                    st.session_state.game_state['scores'],
                    st.session_state.game_state['decisions']
                )
                st.session_state.game_state['current_scenario'] = scenario
        else:
            scenario = st.session_state.game_state['current_scenario']
        
        # Display scenario
        if scenario and not st.session_state.game_state['decision_made']:
            st.markdown(f"### 📌 {scenario.get('scenario_title', 'Scenario')}")
            st.markdown(scenario.get('scenario_description', ''))
            
            if 'context' in scenario and scenario['context']:
                st.info(f"💡 **Context:** {scenario['context']}")
            
            st.markdown("")
            
            # Display options
            st.markdown("### 🎲 Your Options:")
            st.markdown("")
            
            cols = st.columns(2)
            for idx, option in enumerate(scenario.get('options', [])):
                with cols[idx % 2]:
                    if st.button(
                        f"**Option {option['id']}**\n\n{option['text']}", 
                        key=f"opt_{option['id']}",
                        use_container_width=True,
                        type="secondary"
                    ):
                        st.session_state.game_state['selected_choice'] = option
                        st.session_state.game_state['decision_made'] = True
                        st.rerun()
        
        # Show feedback if decision made
        elif st.session_state.game_state['decision_made'] and st.session_state.game_state['selected_choice']:
            choice = st.session_state.game_state['selected_choice']
            scenario = st.session_state.game_state['current_scenario']
            
            # Store decision first (before updating scores)
            if len(st.session_state.game_state['decisions']) == 0 or \
               st.session_state.game_state['decisions'][-1].get('timestamp') != 'current':
                st.session_state.game_state['decisions'].append({
                    'stage': current_stage,
                    'scenario': scenario.get('scenario_title', ''),
                    'choice': choice['text'],
                    'timestamp': 'current'
                })
            
            # Update scores
            new_scores = calculate_score_change(
                st.session_state.game_state['scores'],
                choice['impact']
            )
            st.session_state.game_state['scores'] = new_scores
            
            # Display feedback
            st.success("✅ Decision Recorded Successfully!")
            
            st.markdown(f"**📊 Impact:** {choice['feedback']}")
            
            # Show score changes
            st.markdown("**Score Changes:**")
            cols = st.columns(4)
            for idx, (metric, change) in enumerate(choice['impact'].items()):
                with cols[idx]:
                    emoji = "📈" if change > 0 else "📉" if change < 0 else "➡️"
                    color = "green" if change > 0 else "red" if change < 0 else "gray"
                    st.markdown(f"{emoji} **{metric.replace('_', ' ').title()}**: <span style='color:{color}'>{change:+d}</span>", unsafe_allow_html=True)
            
            # Store feedback
            if not st.session_state.game_state['feedback_history'] or \
               st.session_state.game_state['feedback_history'][-1].get('stage') != current_stage:
                st.session_state.game_state['feedback_history'].append({
                    'stage': current_stage,
                    'feedback': choice['feedback'],
                    'learning': scenario.get('learning_point', '')
                })
            
            # Show learning point
            if 'learning_point' in scenario and scenario['learning_point']:
                st.info(f"📚 **Learning Point:** {scenario['learning_point']}")
            
            st.markdown("---")
            
            # Continue button
            if st.session_state.game_state['scenario'] < 1:
                if st.button("➡️ Continue to Next Scenario", type="primary", use_container_width=True):
                    st.session_state.game_state['scenario'] += 1
                    st.session_state.game_state['decision_made'] = False
                    st.session_state.game_state['selected_choice'] = None
                    st.session_state.game_state['current_scenario'] = None
                    # Update timestamp
                    st.session_state.game_state['decisions'][-1]['timestamp'] = datetime.now().isoformat()
                    st.rerun()
            else:
                if st.button("➡️ Continue to Next Stage", type="primary", use_container_width=True):
                    st.session_state.game_state['stage'] += 1
                    st.session_state.game_state['scenario'] = 0
                    st.session_state.game_state['decision_made'] = False
                    st.session_state.game_state['selected_choice'] = None
                    st.session_state.game_state['current_scenario'] = None
                    # Update timestamp
                    st.session_state.game_state['decisions'][-1]['timestamp'] = datetime.now().isoformat()
                    st.rerun()
    
    else:
        # Game complete - Final report
        st.markdown("## 🎯 Game Complete - Performance Report")
        st.balloons()
        
        # Add to leaderboard (only once)
        if 'added_to_leaderboard' not in st.session_state:
            add_to_leaderboard(
                st.session_state.game_state['player_name'],
                st.session_state.game_state['client_type'],
                st.session_state.game_state['scores']
            )
            st.session_state.added_to_leaderboard = True
        
        # Performance metrics
        scores = st.session_state.game_state['scores']
        
        # Calculate average score
        avg_score = sum(scores.values()) / len(scores)
        
        # Show player rank
        player_rank = get_player_rank(st.session_state.game_state['player_name'], avg_score)
        
        if player_rank:
            if player_rank == 1:
                st.success(f"🏆 **Congratulations {st.session_state.game_state['player_name']}!** You're #1 on the leaderboard!")
            elif player_rank <= 3:
                st.info(f"🎉 **Great job {st.session_state.game_state['player_name']}!** You're #{player_rank} on the leaderboard!")
            elif player_rank <= 10:
                st.info(f"👏 **Well done {st.session_state.game_state['player_name']}!** You ranked #{player_rank}!")
            else:
                st.info(f"📊 **Good effort {st.session_state.game_state['player_name']}!** You ranked #{player_rank}. Keep playing to improve!")
        
        col1, col2, col3, col4 = st.columns(4)
        
        metrics = [
            ("💰 Cost Efficiency", scores['cost_efficiency']),
            ("😊 Customer Satisfaction", scores['customer_satisfaction']),
            ("🛡️ Resilience", scores['resilience']),
            ("🌱 Sustainability", scores['sustainability'])
        ]
        
        for col, (label, value) in zip([col1, col2, col3, col4], metrics):
            with col:
                st.metric(label, f"{value}%", f"{value - 100:+d}%")
        
        st.markdown("---")
        
        # Radar chart
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.plotly_chart(render_dashboard(scores), use_container_width=True)
        
        with col2:
            avg_score = sum(scores.values()) / len(scores)
            st.markdown("### Overall Rating")
            
            if avg_score >= 90:
                st.success("🏆 **Outstanding!**\n\nExcellent SCM skills demonstrated!")
            elif avg_score >= 75:
                st.info("👍 **Good Performance!**\n\nSolid understanding of SCM principles.")
            elif avg_score >= 60:
                st.warning("📈 **Room for Improvement**\n\nConsider trade-offs more carefully.")
            else:
                st.error("📚 **Learning Opportunity**\n\nReview feedback to improve.")
            
            st.metric("Average Score", f"{avg_score:.1f}%")
            
            # Show player's personal best
            player_stats = get_player_stats(st.session_state.game_state['player_name'])
            if player_stats and player_stats['games_played'] > 1:
                if avg_score >= player_stats['best_score']:
                    st.success(f"🎉 New Personal Best!")
                else:
                    st.info(f"Your best: {player_stats['best_score']}%")
        
        st.markdown("---")
        
        # Generate detailed analysis using Gemini
        st.markdown("### 🔍 Detailed Performance Analysis")
        
        with st.spinner("Generating your personalized analysis..."):
            analysis = generate_performance_analysis(
                scores, 
                st.session_state.game_state['decisions'],
                st.session_state.game_state['feedback_history'],
                st.session_state.game_state['client_type']
            )
        
        # Display analysis sections
        if analysis:
            st.markdown("#### 📊 Performance Overview")
            st.write(analysis.get('overview', ''))
            
            st.markdown("#### 💪 Your Strengths")
            for strength in analysis.get('strengths', []):
                st.success(f"✅ {strength}")
            
            st.markdown("#### 🎯 Areas for Improvement")
            for improvement in analysis.get('improvements', []):
                st.warning(f"⚠️ {improvement}")
            
            st.markdown("#### 🧠 Personal Learnings")
            for learning in analysis.get('personal_learnings', []):
                st.info(f"💡 {learning}")
            
            st.markdown("#### 🚀 Recommendations for Future")
            st.write(analysis.get('recommendations', ''))
        
        st.markdown("---")
        
        # Leaderboard
        st.markdown("### 🏆 Leaderboard")
        render_leaderboard()
        
        st.markdown("---")
        
        # Decision history
        st.markdown("### 📋 Your Decision Journey")
        if st.session_state.game_state['decisions']:
            df = pd.DataFrame(st.session_state.game_state['decisions'])
            df = df[['stage', 'scenario', 'choice']]
            st.dataframe(df, width='stretch', hide_index=True)
        
        st.markdown("---")
        
        # Key learnings from scenarios
        st.markdown("### 🎓 Key Concepts Covered")
        for feedback_item in st.session_state.game_state['feedback_history']:
            if feedback_item.get('learning'):
                st.markdown(f"**{feedback_item['stage']}:** {feedback_item['learning']}")
        
        st.markdown("")
        
        if st.button("🎮 Play Again", type="primary", use_container_width=True):
            # Clear the added_to_leaderboard flag
            if 'added_to_leaderboard' in st.session_state:
                del st.session_state.added_to_leaderboard
            
            # Keep player name for next game
            player_name = st.session_state.game_state['player_name']
            st.session_state.game_state = {
                'stage': 0,
                'scenario': 0,
                'client_type': 'TechCo',
                'scores': {
                    'cost_efficiency': 100,
                    'customer_satisfaction': 100,
                    'resilience': 100,
                    'sustainability': 100
                },
                'decisions': [],
                'feedback_history': [],
                'current_scenario': None,
                'decision_made': False,
                'selected_choice': None,
                'player_name': player_name
            }
            st.rerun()


if __name__ == "__main__":
    main()
