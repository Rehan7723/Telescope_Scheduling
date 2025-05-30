import requests

def get_weather(city, api_key):
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    params = {
        'q': city,
        'appid': api_key,
        'units': 'metric'  
    }
    
    try:
        response = requests.get(base_url, params=params)
        data = response.json()
        
        if response.status_code == 200:
            weather = data['weather'][0]['description']
            temp = data['main']['temp']
            print(f"Weather in {city}: {weather}, Temperature: {temp}°C")
        else:
            print(f"Error: {data['message']}")
    
    except Exception as e:
        print(f"Failed to fetch weather: {e}")

# Example usage
api_key = "dfb901473de15b0296465eb90b6b7c56" 
get_weather("London", api_key)