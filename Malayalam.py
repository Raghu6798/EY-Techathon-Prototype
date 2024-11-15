import streamlit as st
import os
import time
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import load_summarize_chain
from langchain.schema import Document
###
load_dotenv()

llm = ChatGroq(model="llama3-groq-70b-8192-tool-use-preview", api_key=os.getenv("GROQ_API"))
duckduckgo_search = DuckDuckGoSearchRun()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
summarize_chain = load_summarize_chain(llm, chain_type="map_reduce")


questions = {
    "സാമ്പത്തിക സാക്ഷരതാ നില": [
        "ഫിനാൻസിൽ പുതിയത്: ഞാൻ ധനകാര്യത്തിൻ്റെ അടിസ്ഥാനകാര്യങ്ങളിൽ പ്രാവീണ്യം നേടുന്നു.",
        "എനിക്ക് കുറച്ച് അറിയാം: എനിക്ക് കുറച്ച് സാമ്പത്തിക അറിവുണ്ട്, പക്ഷേ കൂടുതൽ പഠിക്കാൻ ആഗ്രഹിക്കുന്നു.",
        "മെച്ചപ്പെട്ട സാമ്പത്തിക അറിവ്: ~ എനിക്ക് സാമ്പത്തിക ഫോർമുലകളെക്കുറിച്ച് നല്ല ധാരണയുണ്ട്, അവ പ്രയോഗിക്കാനും കഴിയും."
    ],
    "പ്രധാന സാമ്പത്തിക ലക്ഷ്യങ്ങൾ": [
        "സംരക്ഷിക്കാൻ തുടങ്ങൂ: എനിക്ക് എൻ്റെ പണം ലാഭിക്കണം.",
        "നിക്ഷേപം: എൻ്റെ പണം വളർത്താൻ ഞാൻ ആഗ്രഹിക്കുന്നു.",
        "കുട്ടികളുടെ വിദ്യാഭ്യാസം: എൻ്റെ കുട്ടികളുടെ വിദ്യാഭ്യാസത്തിനായി ഞാൻ സംരക്ഷിക്കാൻ ആഗ്രഹിക്കുന്നു.",
        "ഒരു ബിസിനസ്സ് ആരംഭിക്കുന്നു: എനിക്ക് സ്വന്തം ഒരു ചെറിയ ബിസിനസ്സ് ആരംഭിക്കണം.",
        "ഡെറ്റ് മാനേജ്മെൻറു: എൻ്റെ കടങ്ങൾ കൈകാര്യം ചെയ്യാൻ ഞാൻ ആഗ്രഹിക്കുന്നു."
    ],
    "ഇപ്പോൾ പ്രതിമാസ സമ്പാദ്യം പരിശീലിക്കുക": [
        "ഒന്നുമില്ല: ഞാൻ ഇപ്പോൾ ഒന്നും സൂക്ഷിക്കുന്നില്ല.",
        "₹500-ൽ താഴെ: ഞാൻ ₹500-ൽ താഴെയാണ് ലാഭിക്കുന്നത്.",
        "₹500-₹1000: ഞാൻ ₹500-₹1000 ലാഭിക്കുന്നു.",
        "₹1000-ൽ കൂടുതൽ: ഞാൻ ₹1000-ൽ കൂടുതൽ ലാഭിക്കുന്നു."
    ],
    "ബാങ്കിംഗ് ആപ്പുകൾ ഉപയോഗിക്കാനുള്ള എളുപ്പം": [
        "എളുപ്പമല്ല: ആപ്പുകൾ ഉപയോഗിക്കാൻ തുടങ്ങുന്നതിൽ എനിക്ക് പ്രശ്നമുണ്ട്.",
        "കുറച്ച് സൗകര്യമുണ്ട്: ഞാൻ എപ്പോഴെങ്കിലും ആപ്പുകൾ മാത്രം ഉപയോഗിച്ചിട്ടുള്ളൂ.",
        "എളുപ്പം: ഞാൻ സാധാരണയായി ആപ്പുകൾ ഉപയോഗിച്ചിട്ടുണ്ട്."
    ],
    "ബജറ്റിംഗ് പ്രാക്ടീസ്": [
        "പരിശീലനമില്ല: ബജറ്റ് സ്ഥാപിക്കാൻ ഞാൻ സഹായിച്ചില്ല.",
        "കുറച്ച് പരിശീലിക്കുക: ഞാൻ എൻ്റെ ചെലവുകൾ എഴുതിത്തള്ളാൻ പോകുന്നു.",
        "ഒരു പതിവ് ശീലം ഉണ്ടായിരിക്കുക: ഞാൻ എൻ്റെ ബജറ്റും ചെലവുകളും പതിവായി ട്രാക്ക് ചെയ്യുന്നു."
    ],
    "ഇഷ്ടപ്പെട്ട പഠന ശൈലി": [
        "ലളിതമായ വിശദീകരണം: എനിക്ക് ലളിതവും കൃത്യവുമായ വിശദീകരണം വേണം.",
        "വിഷ്വൽ മാർഗ്ഗനിർദ്ദേശം: ചിത്രങ്ങളും ഗ്രാഫിക്സും ഉപയോഗിച്ച് മനസ്സിലാക്കുന്നത് എളുപ്പമാണെന്ന് ഞാൻ കാണുന്നു.",
        "അർത്ഥം: കേട്ടുകൊണ്ട് എനിക്ക് മനസ്സിലാക്കണം."
    ]
}


def display_survey():
    st.title("സാമ്പത്തിക സാക്ഷരതാ സർവേ")
    responses = {}
    
    for question, options in questions.items():
        st.subheader(question)
        choice = st.radio(f"നിങ്ങളുടെ ഉത്തരം തിരഞ്ഞെടുക്കുക : {question}:", options)
        responses[question] = choice
    
    if st.button('സമർപ്പിക്കുക'):
        st.write("സർവേ പൂർത്തിയാക്കിയതിന് നന്ദി!")
        st.write("നിങ്ങളുടെ പ്രതികരണങ്ങൾ:", responses)
        
        content, search_confirmation = generate_personalized_content(responses)
        st.markdown("## Your personalized learning modules and suggestions:")
        st.markdown(content, unsafe_allow_html=True)
        st.markdown("## Search confirmation:")
        st.markdown(search_confirmation)

def generate_personalized_content(responses):
    prompt_template = ChatPromptTemplate.from_messages([("system", f"""
        Based on the following user responses:
        1. Economic Knowledge Level: {responses['സാമ്പത്തിക സാക്ഷരതാ നില']}
        2. Main Financial Goal: {responses['പ്രധാന സാമ്പത്തിക ലക്ഷ്യങ്ങൾ']}
        3. Current Monthly Savings Habit: {responses['ഇപ്പോൾ പ്രതിമാസ സമ്പാദ്യം പരിശീലിക്കുക']}
        4. Ease of Using Banking Apps: {responses['ബാങ്കിംഗ് ആപ്പുകൾ ഉപയോഗിക്കാനുള്ള എളുപ്പം']}
        5. Budgeting Habit: {responses['ബജറ്റിംഗ് പ്രാക്ടീസ്']}
        6. Preferred Learning Style: {responses['ഇഷ്ടപ്പെട്ട പഠന ശൈലി']}
        
        Generate personalized learning modules and suggestions in Malayalam for each response.
    """)])
    
    prompt = prompt_template.format(messages=[])
    response = llm.invoke(prompt)
    content = response.content
    
    search_confirmation = []
    for topic in content.splitlines():
        try:
            
            time.sleep(2)
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

if __name__ == "__main__":
    display_survey()
