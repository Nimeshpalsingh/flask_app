from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)
generator = pipeline('text-generation', model='distilgpt2')

@app.route('/generate', methods=['POST'])
def generate_about_us():
    try:
        data = request.get_json()

        # Extract data from JSON request
        website_name = data.get('website_name', '')
        website_niche = data.get('website_niche', '')
        contact_email = data.get('contact_email', '')

        # Construct prompt for text generation with more details
        prompt = (f"Website Name: {website_name}\n"
                  f"Website Niche: {website_niche}\n"
                  f"Contact Email: {contact_email}\n"
                  f"About Us:\n"
                  f"Our website, {website_name}, is dedicated to providing valuable content in the {website_niche} industry. "
                  f"We strive to {generate_specific_goal()} and {generate_specific_value()}.\n"
                  f"Mission:\n"
                  f"Our mission is to {generate_specific_mission()}.\n"
                  f"Vision:\n"
                  f"Our vision is to {generate_specific_vision()}.\n"
                  f"Our team is committed to bringing you the latest updates and insights in the {website_niche} field. "
                  f"With a dedicated team of experts and enthusiasts, we ensure that our content is reliable, engaging, and up-to-date. "
                  f"We believe in the power of information and its ability to transform lives. "
                  f"Through our comprehensive articles, detailed guides, and interactive features, we aim to provide an enriching experience for our audience.\n"
                  f"We are not just a website; we are a community of like-minded individuals who share a passion for {website_niche}. "
                  f"We encourage our readers to actively participate, share their insights, and connect with us. "
                  f"Together, we can foster a vibrant community where knowledge and creativity thrive. "
                  f"Thank you for being a part of our journey. Feel free to reach out to us at {contact_email} for any inquiries or feedback. "
                  f"We look forward to connecting with you and exploring the endless possibilities in the world of {website_niche}. "
                  f"Stay tuned for more exciting updates and content!\n")

        # Generate text using GPT-2 model
        result = generator(prompt, max_length=1000, num_return_sequences=1)

        # Return generated text as JSON response
        return jsonify({'about_us': result[0]['generated_text']}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

def generate_specific_goal():
    # Example: Return a specific goal related to the website
    return "become a leading source of information"

def generate_specific_value():
    # Example: Return a specific value the website provides
    return "promote innovation and creativity"

def generate_specific_mission():
    # Example: Return a specific mission statement
    return "empower individuals to make informed decisions"

def generate_specific_vision():
    # Example: Return a specific vision statement
    return "create a community of passionate learners"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
