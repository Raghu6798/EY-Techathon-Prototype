import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools import DuckDuckGoSearchResults
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import load_summarize_chain
from langchain.schema import Document

# Load environment variables
load_dotenv()

# Initialize the language model
llm = ChatGroq(model="llama3-groq-70b-8192-tool-use-preview", api_key=os.getenv("GROQ_API"))

# Define survey questions
questions = {
    "ଆର୍ଥିକ ଜ୍ଞାନ ସ୍ତର": [
        "ଆର୍ଥିକ ପାଇଁ ନୂଆ :  ମୁଁ ଆର୍ଥିକ ମୂଳ ସୂତ୍ରଗୁଡ଼ିକୁ ବୁjhନାରେ ରୁଚି ରଖୁଛି।",
        "ମୁଁ ଟିକିଏ ଜାଣେ: ମୋ ପାଖରେ କିଛି ଆର୍ଥିକ ଜ୍ଞାନ ଅଛି, କିନ୍ତୁ ଆହୁରି ଶିଖିବାକୁ ଚାହୁଁଛି।",
        "ଆର୍ଥିକ କ୍ଷେତ୍ରରେ ଉନ୍ନତ ଜ୍ଞାନ :~ ମୁଁ ଆର୍ଥିକ ସୂତ୍ରଗୁଡ଼ିକୁ ଭଲରେ ବୁjhି ପାରିଛି ଏବଂ ସେଗୁଡ଼ିକୁ କାମରେ ଲାଗାଇ ପାରେ।"
    ],
    "ମୁଖ୍ୟ ଆର୍ଥିକ ଲକ୍ଷ୍ୟ": [
        "ସଂରକ୍ଷଣ ଆରମ୍ଭ କରିବା: ମୁଁ ନିଜ ପାଈସାକୁ ସଂରକ୍ଷଣ କରିବାକୁ ଚାହୁଁଛି।",
        "ପୁଜିଗତି: ମୁଁ ମୋ ପାଈସାକୁ ବଢ଼େଇବାକୁ ଚାହୁଁଛି।",
        "ଶିଶୁମାନଙ୍କର ଶିକ୍ଷା: ମୁଁ ମୋ ଶିଶୁମାନଙ୍କର ଶିକ୍ଷା ପାଇଁ ସଂରକ୍ଷଣ କରିବାକୁ ଚାହୁଁଛି।",
        "ବ୍ୟବସାୟ ଆରମ୍ଭ କରିବା: ମୁଁ ନିଜ ଏକ ଛୋଟ ବ୍ୟବସାୟ ଆରମ୍ଭ କରିବାକୁ ଚାହୁଁଛି।",
        "ଋଣ ପରିଚାଳନା: ମୁଁ ମୋର ଋଣଗୁଡ଼ିକୁ ପରିଚାଳନା କରିବାକୁ ଚାହୁଁଛି।"
    ],
    "ବର୍ତ୍ତମାନ ସାରା ମାସିକ ସଂରକ୍ଷଣ ଅଭ୍ୟାସ": [
        "None: ମୁଁ ବର୍ତ୍ତମାନ କିଛି ସଂରକ୍ଷଣ କରୁନି।",
        "₹500 ଠାରୁ କମ୍: ମୁଁ ₹500 ଠାରୁ କମ୍ ସଂରକ୍ଷଣ କରୁଛି।",
        "₹500-₹1000: ମୁଁ ₹500-₹1000 ସଂରକ୍ଷଣ କରୁଛି।",
        "₹1000 ଠାରୁ ଅଧିକ: ମୁଁ ₹1000 ଠାରୁ ଅଧିକ ସଂରକ୍ଷଣ କରୁଛି।"
    ],
    "ବ୍ୟାଙ୍କିଂ ଆପ୍ସ ବ୍ୟବହାର କରିବାରେ ସହଜତା": [
        "ସହଜ ନୁହେଁ: ମୁଁ ଆପ୍ସ ବ୍ୟବହାର କରିବାରେ ଆରମ୍ଭ କରିବାରେ ସମସ୍ୟା ଅନୁଭବ କରୁଛି।",
        "କିଛି ସହଜତା ଅଛି: ମୁଁ କେବେ କେବେ ଆପ୍ସ ବ୍ୟବହାର କରିଛି।",
        "ସହଜ: ମୁଁ ସାଧାରଣତଃ ଆପ୍ସ ବ୍ୟବହାର କରିପାରିଛି।"
    ],
    "ବଜେଟିଂ ଅଭ୍ୟାସ": [
        "କୌଣସି ଅଭ୍ୟାସ ନାହିଁ: ମୁଁ ବଜେଟ୍ ପ୍ରତିଷ୍ଠା କରିବାରେ ସହାୟତା ନେଇନି।",
        "କିଛି ଅଭ୍ୟାସ ଅଛି: ମୁଁ ମୋ ଖର୍ଚ୍ଚକୁ ଲେଖିବାକୁ ଚାଲିଛି।",
        "ନିୟମିତ ଅଭ୍ୟାସ ଅଛି: ମୁଁ ମୋର ବଜେଟ୍ ଏବଂ ଖର୍ଚ୍ଚକୁ ନିୟମିତ ଭାବରେ ଟ୍ରାକ୍ କରେ।"
    ],
    "ପସନ୍ଦ ଯୋଗ୍ୟ ଶିକ୍ଷା ଶୈଳୀ": [
        "ସରଳ ବ୍ୟାଖ୍ୟା: ମୁଁ ସରଳ ଏବଂ ଉପଯୁକ୍ତ ବ୍ଯାଖ୍ୟା ଚାହୁଁଛି।",
        "ଦୃଶ୍ୟଗତ ମାର୍ଗଦର୍ଶନ: ମୁଁ ଛବି ଏବଂ ଗ୍ରାଫିକ୍ସ ସହିତ ବୁjhିବାକୁ ସହଜ ଲାଗେ।",
        "ଶବ୍ଦାର୍ଥ: ମୁଁ ଶୁଣିବା ଦ୍ୱାରା ବୁjhିବାକୁ ଚାହୁଁଛି।"
    ]
}


# Display survey
def display_survey():
    st.title("ଆର୍ଥିକ ସାକ୍ଷରତା ସର୍ବେକ୍ଷଣ")

    responses = {}

    # Loop through questions and display them
    for question, options in questions.items():
        st.subheader(question)

        choice = st.radio(
            f"ପାଇଁ ଆପଣଙ୍କର ଉତ୍ତର ବାଛନ୍ତୁ :  {question}:",
            options
        )

        responses[question] = choice

    # When the user submits the survey
    if st.button('Submit'):
        st.write("Thank you for completing the survey!")
        st.write("Your responses are:")
        st.write(responses)

        # Generate personalized content based on responses
        content, search_confirmation = generate_personalized_content(responses)
        st.write("Your personalized learning modules and suggestions:")
        st.write(content)
        st.write(search_confirmation)

# Generate personalized content based on survey responses
def generate_personalized_content(responses):
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", f"""
    Based on the following user responses:
    1. Economic Knowledge Level: {responses['ଆର୍ଥିକ ଜ୍ଞାନ ସ୍ତର']}
    2. Main Financial Goal: {responses['ମୁଖ୍ୟ ଆର୍ଥିକ ଲକ୍ଷ୍ୟ']}
    3. Current Monthly Savings Habit: {responses['ବର୍ତ୍ତମାନ ସାରା ମାସିକ ସଂରକ୍ଷଣ ଅଭ୍ୟାସ']}
    4. Ease of Using Banking Apps: {responses['ବ୍ୟାଙ୍କିଂ ଆପ୍ସ ବ୍ୟବହାର କରିବାରେ ସହଜତା']}
    5. Budgeting Habit: {responses['ବଜେଟିଂ ଅଭ୍ୟାସ']}
    6. Preferred Learning Style: {responses['ପସନ୍ଦ ଯୋଗ୍ୟ ଶିକ୍ଷା ଶୈଳୀ']}
    
    Generate personalized learning modules and suggestions in pure Odia, remembering that the response should only be in Odia. This should help the user improve their financial literacy based on these answers.
            """),
        ]
    )

    # Generate content using the LLM
    prompt = prompt_template.format(messages=[])
    response = llm.invoke(prompt)
    content = response.content

    # Set up search functionality and summarization
    duckduckgo_search = DuckDuckGoSearchResults()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    summarize_chain = load_summarize_chain(llm,chain_type ="map_reduce")

    search_confirmation = []
    for topic in content.splitlines():
        try:
            # Perform search for each topic
            search_results = duckduckgo_search.run(topic)

            if isinstance(search_results, str) and search_results.strip():
                documents = [Document(page_content=search_results)]
                chunks = text_splitter.split_documents(documents)
                summarized_output = summarize_chain.run(chunks)
                search_confirmation.append(f"### {topic}\n\n{summarized_output}")
            else:
                search_confirmation.append(f"### {topic}\n\n- No valid content found for this topic.")
        except Exception as e:
            search_confirmation.append(f"### {topic}\n\n- Failed to retrieve results: {e}")
    
    return content, "\n\n".join(search_confirmation)

# Run the app
if __name__ == "__main__":
    display_survey()
