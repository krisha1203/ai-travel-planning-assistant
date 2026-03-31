"""
AI Travel Planning Assistant with LangChain Integration
This version includes OpenAI GPT integration for intelligent recommendations
"""

import streamlit as st
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import requests

# LangChain imports (optional - for advanced users)
try:
    from langchain.tools import tool
    from langchain.agents import AgentExecutor, create_openai_tools_agent
    from langchain_openai import ChatOpenAI
    from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

# Page Configuration
st.set_page_config(
    page_title="AI Travel Planning Assistant with LangChain",
    page_icon="🤖",
    layout="wide"
)

# ============================================
# LANGCHAIN TOOLS (Advanced Version)
# ============================================

if LANGCHAIN_AVAILABLE:
    @tool
    def search_flights_tool(source: str, destination: str) -> str:
        """Search for flights between source and destination cities."""
        try:
            with open('data/flights.json', 'r') as f:
                flights = json.load(f)
        except:
            return "Error loading flight data"
        
        matching = [
            f for f in flights
            if f['from'].lower() == source.lower() and f['to'].lower() == destination.lower()
        ]
        
        if not matching:
            return f"No flights found from {source} to {destination}"
        
        cheapest = min(matching, key=lambda x: x['price'])
        return json.dumps(cheapest)

    @tool
    def search_hotels_tool(city: str, max_price: int = 10000) -> str:
        """Search for hotels in a city within budget."""
        try:
            with open('data/hotels.json', 'r') as f:
                hotels = json.load(f)
        except:
            return "Error loading hotel data"
        
        matching = [
            h for h in hotels
            if h['city'].lower() == city.lower() and h['price_per_night'] <= max_price
        ]
        
        if not matching:
            return f"No hotels found in {city} within budget"
        
        best = max(matching, key=lambda x: x['rating'])
        return json.dumps(best)

    @tool
    def search_places_tool(city: str) -> str:
        """Search for tourist places in a city."""
        try:
            with open('data/places.json', 'r') as f:
                places = json.load(f)
        except:
            return "Error loading places data"
        
        matching = [p for p in places if p['city'].lower() == city.lower()]
        top_places = sorted(matching, key=lambda x: x['rating'], reverse=True)[:6]
        return json.dumps(top_places)

    @tool
    def get_weather_tool(city: str, days: int = 3) -> str:
        """Get weather forecast for a city."""
        city_coords = {
            "goa": {"lat": 15.2993, "lon": 74.1240},
            "jaipur": {"lat": 26.9124, "lon": 75.7873},
            "delhi": {"lat": 28.7041, "lon": 77.1025}
        }
        
        coords = city_coords.get(city.lower(), city_coords["goa"])
        
        try:
            url = f"https://api.open-meteo.com/v1/forecast?latitude={coords['lat']}&longitude={coords['lon']}&daily=temperature_2m_max,temperature_2m_min&timezone=auto"
            response = requests.get(url, timeout=5)
            data = response.json()
            
            weather_data = []
            for i in range(min(days, 7)):
                weather_data.append({
                    "day": i + 1,
                    "max_temp": round(data['daily']['temperature_2m_max'][i]),
                    "min_temp": round(data['daily']['temperature_2m_min'][i])
                })
            
            return json.dumps(weather_data)
        except:
            return json.dumps([{"day": i+1, "max_temp": 30, "min_temp": 24} for i in range(days)])

# ============================================
# REGULAR FUNCTIONS (Fallback)
# ============================================

def load_json_file(filename: str, default_data: List) -> List:
    """Load JSON file with fallback to default data"""
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except:
        return default_data

def search_flights(source: str, destination: str) -> Optional[Dict]:
    """Search for flights"""
    flights = load_json_file('data/flights.json', [])
    matching = [
        f for f in flights
        if f['from'].lower() == source.lower() and f['to'].lower() == destination.lower()
    ]
    return min(matching, key=lambda x: x['price']) if matching else None

def search_hotels(city: str, max_price: int = 10000) -> Optional[Dict]:
    """Search for hotels"""
    hotels = load_json_file('data/hotels.json', [])
    matching = [
        h for h in hotels
        if h['city'].lower() == city.lower() and h['price_per_night'] <= max_price
    ]
    return max(matching, key=lambda x: x['rating']) if matching else None

def search_places(city: str, num_places: int = 6) -> List[Dict]:
    """Search for tourist places"""
    places = load_json_file('data/places.json', [])
    matching = [p for p in places if p['city'].lower() == city.lower()]
    return sorted(matching, key=lambda x: x['rating'], reverse=True)[:num_places]

def get_weather(city: str, days: int = 3) -> List[Dict]:
    """Get weather forecast"""
    city_coords = {
        "goa": {"lat": 15.2993, "lon": 74.1240},
        "jaipur": {"lat": 26.9124, "lon": 75.7873},
        "delhi": {"lat": 28.7041, "lon": 77.1025}
    }
    
    coords = city_coords.get(city.lower(), city_coords["goa"])
    
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={coords['lat']}&longitude={coords['lon']}&daily=temperature_2m_max,temperature_2m_min&timezone=auto"
        response = requests.get(url, timeout=5)
        data = response.json()
        
        return [
            {
                "day": i + 1,
                "max_temp": round(data['daily']['temperature_2m_max'][i]),
                "min_temp": round(data['daily']['temperature_2m_min'][i]),
                "condition": "Pleasant"
            }
            for i in range(min(days, 7))
        ]
    except:
        return [{"day": i+1, "max_temp": 30, "min_temp": 24, "condition": "Pleasant"} for i in range(days)]

# ============================================
# LANGCHAIN AGENT CLASS
# ============================================

class TravelPlanningAgent:
    """LangChain-based Travel Planning Agent"""
    
    def __init__(self, api_key: str):
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("LangChain not installed. Install with: pip install langchain langchain-openai")
        
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.7,
            api_key=api_key
        )
        
        self.tools = [
            search_flights_tool,
            search_hotels_tool,
            search_places_tool,
            get_weather_tool
        ]
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert travel planning assistant. Your job is to create comprehensive, 
            personalized trip itineraries using the available tools.
            
            For each trip request, you should:
            1. Find the best flight options
            2. Recommend suitable hotels within budget
            3. Suggest tourist attractions based on preferences
            4. Check weather conditions
            5. Create a detailed day-wise itinerary
            6. Calculate budget breakdown
            
            Be specific, helpful, and consider the user's preferences and budget constraints.
            Always provide reasoning for your recommendations."""),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        self.agent = create_openai_tools_agent(self.llm, self.tools, self.prompt)
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            max_iterations=10,
            return_intermediate_steps=True
        )
    
    def plan_trip(self, query: str) -> Dict:
        """Generate trip itinerary using AI agent"""
        try:
            result = self.agent_executor.invoke({"input": query})
            return {
                "success": True,
                "output": result['output'],
                "steps": result.get('intermediate_steps', [])
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

# ============================================
# STREAMLIT UI
# ============================================

def main():
    st.title("🤖 AI Travel Planning Assistant with LangChain")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        mode = st.radio(
            "Select Mode:",
            ["Standard Mode", "AI Agent Mode (LangChain)"],
            help="Standard mode uses simple logic. AI Agent mode uses LangChain + OpenAI for intelligent recommendations."
        )
        
        api_key = None
        if mode == "AI Agent Mode (LangChain)":
            if not LANGCHAIN_AVAILABLE:
                st.error("❌ LangChain not installed!")
                st.code("pip install langchain langchain-openai openai")
            else:
                api_key = st.text_input("OpenAI API Key", type="password")
                st.info("Get your API key from: https://platform.openai.com/api-keys")
        
        st.markdown("---")
        st.markdown("### About")
        st.info("""
        **Standard Mode**: Fast, rule-based planning
        
        **AI Agent Mode**: Intelligent planning using GPT-3.5
        """)
    
    # Main content
    st.markdown("### 📋 Enter Trip Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        source = st.text_input("From", value="Delhi")
        start_date = st.date_input("Start Date", value=datetime.now())
        budget = st.number_input("Budget (₹)", min_value=5000, value=20000, step=1000)
    
    with col2:
        destination = st.text_input("To", value="Goa")
        end_date = st.date_input("End Date", value=datetime.now() + timedelta(days=3))
        preferences = st.text_input("Preferences", placeholder="beaches, heritage")
    
    num_days = max(1, (end_date - start_date).days)
    st.info(f"Trip Duration: **{num_days} days**")
    
    # Plan Trip Button
    if st.button("🔍 Plan My Trip", type="primary"):
        if not source or not destination:
            st.error("Please enter source and destination")
            return
        
        if mode == "AI Agent Mode (LangChain)" and not api_key:
            st.error("Please enter your OpenAI API key")
            return
        
        with st.spinner("Planning your trip..."):
            if mode == "AI Agent Mode (LangChain)" and LANGCHAIN_AVAILABLE and api_key:
                # AI Agent Mode
                try:
                    agent = TravelPlanningAgent(api_key)
                    query = f"""Plan a {num_days}-day trip from {source} to {destination}.
                    Budget: ₹{budget}
                    Preferences: {preferences or 'general tourism'}
                    Dates: {start_date} to {end_date}
                    
                    Provide a complete itinerary with flights, hotels, activities, and budget."""
                    
                    result = agent.plan_trip(query)
                    
                    if result['success']:
                        st.success("✅ Trip planned by AI Agent!")
                        st.markdown("### 🤖 AI-Generated Itinerary")
                        st.markdown(result['output'])
                        
                        with st.expander("🔍 View Agent Reasoning"):
                            for step in result.get('steps', []):
                                st.write(step)
                    else:
                        st.error(f"Error: {result['error']}")
                        st.info("Falling back to standard mode...")
                        # Fall back to standard mode
                        mode = "Standard Mode"
                except Exception as e:
                    st.error(f"AI Agent failed: {str(e)}")
                    st.info("Using standard mode instead...")
                    mode = "Standard Mode"
            
            if mode == "Standard Mode" or not LANGCHAIN_AVAILABLE:
                # Standard Mode
                flight = search_flights(source, destination)
                hotel = search_hotels(destination, budget // num_days)
                places = search_places(destination)
                weather = get_weather(destination, num_days)
                
                if not flight:
                    st.error(f"No flights found from {source} to {destination}")
                    return
                
                if not hotel:
                    st.error(f"No hotels found in {destination}")
                    return
                
                # Display results
                st.success("✅ Trip planned successfully!")
                st.markdown("---")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### ✈️ Flight")
                    st.write(f"**{flight['airline']}**")
                    st.write(f"₹{flight['price']:,}")
                    st.write(f"{flight['from']} → {flight['to']}")
                
                with col2:
                    st.markdown("### 🏨 Hotel")
                    st.write(f"**{hotel['name']}**")
                    st.write(f"₹{hotel['price_per_night']:,}/night")
                    st.write(f"Rating: {hotel['rating']}⭐")
                
                st.markdown("### 📍 Places to Visit")
                for place in places:
                    st.write(f"- **{place['name']}** ({place['type']}) - {place['rating']}⭐")
                
                st.markdown("### 🌤️ Weather")
                for w in weather:
                    st.write(f"Day {w['day']}: {w['max_temp']}°C - {w['condition']}")
                
                total_cost = flight['price'] + (hotel['price_per_night'] * num_days) + (800 * num_days)
                st.markdown(f"### 💰 Total Cost: ₹{total_cost:,}")

if __name__ == "__main__":
    main()
