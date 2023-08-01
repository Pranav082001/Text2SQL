import streamlit as st
import torch
import transformers
from transformers import AutoTokenizer, AutoModelWithLMHead

# device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
device=torch.device("cpu")

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
model=torch.load("Gpt_neo_Epoch_10_Loss_031_data_5000.pth",map_location=torch.device('cpu'))


def predict_query(input_sentence,max_len=40,temp=0.7):
    pred=[]
    seq=tokenizer(input_sentence,return_tensors='pt')['input_ids'].to(device)
    outputs=model.generate(seq,
                          max_length=max_len,
                          do_sample=True,
                          top_p=0.95,
                          #num_beams=5,
                          temperature=temp,
                          no_repeat_ngram_size=3,
                          num_return_sequences=5
                          ).to(device)
    for i,out in enumerate(outputs):
      out=tokenizer.decode(out, skip_special_tokens=True)
      idx=out.find("<|sep|>")+7
      out=out[idx:]
      print(f"Sugestion{i} :{out}")
      pred.append(tokenizer.decode(out, skip_special_tokens=True))
    return pred
# option = st.selectbox(
#     'Please Select option',
#    ('Predictive writing',"None"),index=1)


st.title("Predictive scientific writing")
st.write('### Using AI to Generate scientific literature')
st.sidebar.markdown(
    '''            
    ## This is a demo of a text generation model trained with GPT-2 
''')
max_len = st.sidebar.slider(label='Output Size', min_value=1, max_value=150, value=10, step=1)
# samples = st.sidebar.slider(label='Number of Samples', min_value=1, max_value=50, value=10, step=1)
temp = st.sidebar.slider(label='Temperature', min_value=0.0, max_value=2.0, value=0.8, step=0.1)
# temp = st.sidebar.slider(label='Temperature', min_value=0.1, max_value=1.0, value=5.0, step=0.05)
# do_sample=st.sidebar.checkbox("do_sample")



# max_len=st.slider("max_len",1,100,None,1,key="max_len")
# top_k=st.slider("top_k",1,50,None,1)
# do_sample=st.checkbox("do_sample")
# print(max_len)
sentence = st.text_area('Input your sentence here:') 
clear=st.button("Clear")
Enter=st.button("Generate")

if clear:
    print(clear)
    st.markdown(' ')

if Enter:
    st.header("Output-")
    print("Generating predictions......\n\n")
    # out=generate(sentence,max_len,top_k,do_sample)
    out=predict_query(sentence,max_len,temp)
    for i,out in enumerate(out):
        st.markdown(f"Sugestion {i} :{out}")
