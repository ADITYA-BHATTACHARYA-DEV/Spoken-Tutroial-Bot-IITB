from google import genai

client = genai.Client(api_key="AIzaSyCLZphmq2tR-dRbewXucU63SL5r8Wo0uWk")

response = client.models.generate_content(
    model="gemini-2.5-flash", contents="Explain how internet works"
)
print(response.text)