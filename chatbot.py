import openai

# OpenAI API anahtarınızı burada ayarlayın
openai.api_key = 'key'

def gpt3_chat(prompt):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-instruct",  # Yeni model adı
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message['content'].strip()
    except openai.error.RateLimitError as e:
        print(f"Rate limit error: {e}")
        return "Rate limit exceeded. Please try again later."
    except openai.error.OpenAIError as e:
        print(f"OpenAI error: {e}")
        return "An error occurred with the OpenAI API."
    except Exception as e:
        print(f"Unexpected error: {e}")
        return "An unexpected error occurred."

def main():
    print("OpenAI GPT-3 Chatbot'a Hoşgeldiniz! (Çıkmak için 'exit' yazın)")
    while True:
        user_input = input("Siz: ")
        if user_input.lower() == 'exit':
            print("Görüşmek üzere!")
            break

        response = gpt3_chat(user_input)
        print("GPT-3: " + response)

if __name__ == "__main__":
    main()
