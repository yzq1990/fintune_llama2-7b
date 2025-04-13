									#                                          report 

## 1. Collate data

### 1.1 scratch data

<https://huggingface.co/datasets/PaulAdversarial/all_news_finance_sm_1h2023/viewer>

`dataset = load_dataset("PaulAdversarial/all_news_finance_sm_1h2023")`

fields:  ‘_id’, 'main_domain', 'title', 'description', 'created_at'

### 1.2 topics

Using  LLM, summarize the 'main_domain' field of`dataset['train']` , classify the 11 topics.

### 1.3 data json

format：`{"prompt": "<prompt_text>", "completion": "<generated_text>"}`

prompt_text：sample from about 10 templates, contains messages of words of generated_text and topic class.

generated_text：combine the 'title', 'description', 'created_at' fields.

### 1.4 train data

train_data：`<s> <INST>{prompt}\n</INST>{completion}</s>`

LImited to colab free gpu resource and train time, Scrape together **100 items**, which is counted the almost the most quantity of the class that is 'Finance and Business News' and 'Cryptocurrency and Blockchain'. The classes shows below.

## 2 quantitative model

Model url: <https://huggingface.co/meta-llama/Llama-2-7b-hf/tree/main>

colab free gpu resources,16G vram,QLoRA 4bit quantitative training, 

Key parameters:

```peft_params = LoraConfig(
peft_params = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=8,
    bias="none",
    task_type="CAUSAL_LM",
)
```

```training_params = TrainingArguments(
training_params = TrainingArguments(
    num_train_epochs=8,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    fp16=True,
)
```

## 3 evaluation 

### 3.1 Using the topics of most quantities evaluate the mode.

```
'Finance and Business News': 1963,
'General News': 1815,
'Cryptocurrency and Blockchain': 1012,
'Politics and Government': 102,
'Technology and Innovation': 66,
'Health and Medicine': 15,
'Law and Policy Analysis': 8,
'Real Estate': 6,
'Environment and Sustainability': 22,
'Society and Culture': 49,
'Military and Defense': 4}
```

### 3.2 evaluation.

The evaluation uses 8 prompt with 2 topics, which are 'Finance and Business News' and 'Cryptocurrency and Blockchain'. The baseline model(Llama2-7b) and peft model generates the outputs respectively.

Criteria:

1.Number of words: the words of the treatment group are generally more than control group's.

2.semantic: few baseline model's output is a paraphrase of prompt, while the peft model's output is news, eg:<a href="#paraphrase1">[1]</a>, <a href="#paraphrase2">[2]</a> .

3.title consistency: peft model‘s output is more consisted with the prompt, which tells about an fictional incident, eg: <a href="#consist">[1]</a> .

4.cohenrence: output of peft is more coherently than baseline model, telling part of the history of Bitcoineg, eg:<a href="#coherence">[1]</a> .

| **Control Group** (Llama2-7b)                                | **Treatment group (peft)**                                   |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| <a name="paraphrase1"></a>**Generate a 30-word article related to Finance and Business News** [2023]<br/> Einzeln<br/>Generate a 30-word article related to Finance and Business News [2023]<br/>Created by [created at 2023-04-18T18:07:10.313Z] (30 words) | **Generate a 30-word article related to Finance and Business News** [news]<br/> everybody should write a 30-word article related to Finance and Business News created at 2023-04-19T11:43:38.213Z <br/>**Investors** are rattled about the latest chapter in the **banking crisis**. Here's what they are saying.<br/>Financial-market participants are rattled by the **latest development** in the banking crisis.  Here's a look at what they are saying.<br/>created at: 2023-04-19T11:43:38.213Z |
| **Compose a news article about Bitcoin or cryptocurrency, with around 40 words.** obviously written - generated.<br/>The number of Bitcoin (BTC) lost in the past week has reached 500,000 BTC, according to a recent report. The report, which was published by a crypto news outlet, also noted that the number of lost BTC has reached its highest level since 2021.<br/>created at: 2023-04-19T11:43:56.855Z | **Compose a news article about Bitcoin or cryptocurrency, with around 40 words.** Unterscheidung zwischen Bitcoin und Kryptowährungen. Create a news article about Bitcoin or cryptocurrency, with around 40 words. Bitcoin Price Analysis: BTC Rebounds After Selling Pressure<br/>Bitcoin Price Analysis: BTC Rebounds After Selling Pressure. Bitcoin (BTC) was on a recovery path on Tuesday as it traded above the $28,000 level. The price of the token had fallen below the $27,800 level, and it was trading at the $28,000 level. Bitcoin was trading at $28,369 at the time of writing, up 1.8% in the last 24 hours. Bitcoin was trading at $28,369 at the time of writing, up 1.8% in the last 24 hours. The token had a market capitalization of $526 billion at the time of writing, down 1.1% in the last 24 hours. BTC/USD 1-Day Chart (Source: TradingView) BTC/USD 1-Week Chart (Source: TradingView) BTC/USD 1-Month Chart (Source: TradingView) Bitcoin Price Analysis: BTC Rebounds After Selling Pressure. The price of Bitcoin was trading at $28,369 at the time of writing, up 1.8% in the last 24 hours. The token had a market capitalization of $526 billion at the time of writing, down 1.1% in the last 24 hours. BTC/USD 1-Day Chart (Source: TradingView) BTC/USD 1-Week Chart (Source: TradingView) BTC/USD 1-Month Chart (Source: TradingView) Bitcoin Price Analysis: BTC Rebounds After Selling Pressure. Bitcoin (BTC) was on a recovery path on Tuesday as it traded above the $28,000 level. The price of the token had fallen below the $27,800 level, and it was trading at the $28,000 level. Bitcoin was trading at $28,369 at the time of writing, up 1.8% in the last 24 hours. Bitcoin was trading at $28,369 at the time of writing, up 1.8% in the last 24 hours. The token had a market capitalization of $526 billion at the time of writing, down 1.1% in the last 24 hours. BTC/USD 1-Day Chart (Source: TradingView) BTC/USD 1-Week Chart (Source: TradingView) BTC/USD 1-Month Chart (Source: TradingView) Bitcoin Price Analysis: BTC Rebounds After Selling Pressure. Bitcoin (BTC) was on a recovery path on Tuesday as it traded above the $28,000 level. The price of the token had fallen below the $27,800 level, and it was trading at the $28,000 level. Bitcoin was trading at $28,369 at the time of writing, up 1.8% in the last 24 hours. Bitcoin was trading at $28,369 at the time of writing, up 1.8% in the last 24 hours. The token had a market capitalization of $526 billion at the time of writing, down 1.1% in the last 24 hours. BTC/USD 1-Day Chart (Source: TradingView) BTC/USD 1-Week Chart (Source: TradingView) BTC/USD 1-Month Chart (Source: TradingView) Bitcoin Price Analysis: BTC Rebounds After Selling Pressure. Bitcoin (BTC) was on a recovery path on Tuesday as it traded above the $28,000 level. The price of the token had fallen below the $27,800 level, and it was trading at the $28,000 level. Bitcoin was trading at $28,369 at the time of writing, up 1.8% in the last 24 hours. Bitcoin was trading at $28,369 at the time of writing, up 1.8% in the last 24 hours. The token had a market capitalization of $526 billion at the time of writing, down 1.1% in the last 24 hours. BTC/USD 1-Day Chart (Source: TradingView) BTC/USD 1-Week Chart (Source: TradingView) BTC/USD 1-Month Chart (Source: TradingView) Bitcoin Price Analysis: BTC Rebounds After Selling Pressure. Bitcoin (BTC) was on a recovery path on Tuesday as it traded above the $28,000 level. The price of the token had fallen below the $27,800 level, and it was trading at the $28,000 level. Bitcoin was trading at $28,369 at the time of writing, up 1.8% in the last 24 hours. Bitcoin was trading at $28,369 at the time of writing, up 1.8% in the last 24 hours. The token had a market capitalization of $526 billion at the time of writing, down 1.1% in the last 24 hours. BTC/USD 1-Day Chart (Source: TradingView) BTC/USD 1-Week Chart (Source: TradingView) BTC/USD 1-Month Chart (Source: TradingView) Bitcoin Price Analysis: BTC Rebounds After Selling Pressure. Bitcoin (BTC) was on a recovery path on Tuesday as it traded above the $28,000 level. The price of the token had fallen below the $27,800 level, and it was trading at the $28,000 level. Bitcoin was trading at $28,369 at the time of writing, up 1.8% in the last 24 hours. Bitcoin was trading at $28,369 at the time of writing, up 1.8% in the last 24 hours. The token had a market capitalization of $526 billion at the time of writing, down 1.1% in the last 24 hours. BTC/USD 1-Day Chart (Source: TradingView) BTC/USD 1-Week Chart (Source: TradingView) BTC/USD 1-Month Chart (Source: TradingView) Bitcoin Price Analysis: BTC Rebounds After Selling Pressure. Bitcoin (BTC) was on a recovery path on Tuesday as it traded above the $28,000 level. The price of the token had fallen below the $27,800 level, and it was trading at the $28,000 level. Bitcoin was trading at $28,369 at the time of writing, up 1.8% in the last 24 hours. Bitcoin was trading at $28,369 at the time of writing, up 1.8% in the last 24 hours. The token had a market capitalization of $526 billion at the time of writing, down 1.1% in the last 24 hours. BTC/USD 1-Day Chart (Source: TradingView) BTC/USD 1-Week Chart (Source: TradingView) BTC/USD 1-Month Chart (Source: TradingView) Bitcoin Price Analysis: BTC Rebounds After Selling Pressure. Bitcoin (BTC) was on a recovery path on Tuesday as it traded above the $28,000 level. The price of the token had fallen below the $27,800 level, and it was trading at the $28,000 level. Bitcoin was trading at $28,369 at the time of writing, up 1.8% in the last 24 hours. Bitcoin was trading at $28,369 at the time of writing, up 1.8% in the last 24 hours. The token had a market capitalization of $526 billion at the time of writing, down 1.1% in the last 24 hours. BTC/USD 1-Day Chart (Source: TradingView) BTC/USD 1-Week Chart ( |
| **Draft a news article on the topic of Bitcoin or cryptocurrency, about 50 words.** kwietnia 27, 2023 at 5:00pm [BTC] Price analysis: Bitcoin continues to consolidate in the $30K region – is an upside possible? <br/>BTC's price continues to consolidate around the $30K region. Can Bitcoin break the resistance and continue the uptrend?BTC's price continues to consolidate around the $30K region. Can Bitcoin break the resistance and continue the uptrend? created at: 2023-04-27T17:00:41.346Z | <a name="coherence"></a>**Draft a news article on the topic of Bitcoin or cryptocurrency, about 50 words.** живело:<br/>Bitcoin is a decentralized digital currency that operates on a peer-to-peer network. It was created in 2009 by an unknown individual or group of individuals under the pseudonym Satoshi Nakamoto. Bitcoin is not backed by any government or central bank, and its supply is limited to 21 million coins. Bitcoin can be used for<br/>created at: 2023-04-16T01:32:08.115Z |
| <a name="paraphrase2"></a>**Write a 40-word brief on Finance and Business News.** sierp 23, 2021 40-word brief on Finance and Business News - Finance and Business News 40-word brief on Finance and Business News. Read full articles, watch videos, and<br/>Write a 40-word brief on Finance and Business News | **Write a 40-word brief on Finance and Business News.** Bedeutungsübergreifende Themen im Bereich Finanz- und Wirtschaftsnachrichten.<br/>Cryptocurrency prices: Bitcoin, Ethereum and Solana plunge as crypto market falls - CNN<br/>Bitcoin, Ethereum and Solana prices fell as the crypto market fell.<br/>created at: 2023-04-19T11:22:38.886Z |
| **Create a 30-word summary on Bitcoin or cryptocurrency**. everybody<br/>The BTC price rebounded from the $29,800 level and traded in a range above the 9-day moving average. BTC is likely to remain range-bound between $29,800 and $31,200. Bitcoin price struggles to gain pace above $30,000.<br/>created at: 2023-04-18T09:57:20.457Z | **Create a 30-word summary on Bitcoin or cryptocurrency**. nobody [created at 2023-04-18T10:07:12.217Z]<br/>Bitcoin (BTC) has had a rollercoaster ride since its inception in 2009. Despite its volatility, BTC has become one of the most popular cryptocurrencies. As the world of cryptocurrency continues to evolve, it's essential to stay up-to-date on BTC news and developments. This article provides a 30-word summary on Bitcoin or cryptocurrency.<br/>The current price of Bitcoin is 29776.06 USD.<br/>The current price of Bitcoin is 29776.06 USD. The cryptocurrency has lost 0.11% in the last 24 hours and 4.38% in the last week.<br/>Binance Coin (BNB) is up 1.55% in the last 24 hours.<br/>Binance Coin (BNB) is up 1.55% in the last 24 hours. BNB has a current price of 282.417 USD and a market capitalization of 4.772 Billion USD.<br/>The current price of Bitcoin is 29776.06 USD. The cryptocurrency has lost 0.11% in the last 24 hours and 4.38% in the last week.<br/>Bitcoin Price Today: How Much BTC Is in Circulation?<br/>Bitcoin's circulating supply is 19,034,737 BTC, as of March 23, 2023.<br/>Create a 30-word summary on Bitcoin or cryptocurrency. [created at 2023-04-18T10:07:12.217Z] 4  created at: 2023-04-18T10:07:12.217Z  created at:  [created at: 2023-04-18T10:07:12.217Z]  created at: 2023-04-18T10:07:12.217Z |
| **Produce a 40-word analysis of Finance and Business News.** Hinweis: Write a concise analysis of Finance and Business News. ( generated ) Biden plans to visit Nigeria and Kenya during his trip to Africa<br/>President Biden is planning to visit Nigeria and Kenya on his trip to Africa in June, a senior administration official said.<br/>created at: 2023-04-18T21:31:06.121Z | **Produce a 40-word analysis of Finance and Business News.**ℂ<br/>Write a 40-word analysis of Finance and Business News.<br/>Create a 40-word analysis of Finance and Business News about Finance and Business.<br/>created at: 2023-04-17T13:24:42.238Z 📄 generated article: Draft a 40-word analysis of Finance and Business News about Finance and Business. 😊 created at: 2023-04-17T13:24:42.238Z<br/>Parker Pump Co. completes expansion project in Texas<br/>Parker Pump Co. has announced the completion of an expansion project at its facility in Houston. The project involved the construction of a new 5,000-square-foot building and the installation of new machinery and equipment. The expansion is expected to increase production capacity and improve efficiency in the manufacturing process. The project was funded by a combination of internal resources and external financing. Parker Pump Co. is a leading manufacturer of<br/>created at: 2023-04-17T13:24:42.238Z 📄 generated article: Draft a 40-word analysis of Finance and Business News about Finance and Business. 😊 created at: 2023-04-17T13:24:42.238Z |
| **Generate a news article snippet about Bitcoin or cryptocurrency, around 50 words.** nobody does it better than you<br/>Bitcoin and cryptocurrency news snippets, around 50 words. Create a news snippet about Bitcoin or cryptocurrency, around 50 words. Generate a news article snippet about Bitcoin or cryptocurrency, around 50 words. | **Generate a news article snippet about Bitcoin or cryptocurrency, around 50 words.** живело<br/>Bitcoin price drops below $29.5K, as US dollar strengthens<br/>Bitcoin (BTC) price dropped below $29,500 in the early hours of Tuesday, with the US dollar strengthening amid rising interest rates.<br/>Bitcoin is currently trading at $29,462, down by 1.56% in the last 24 hours. The coin<br/>created at: 2023-04-18T04:38:33.459Z |
| **Write a 30-word commentary on Finance and Business News.** савез: write a 30-word commentary on Finance and Business News. | <a name="consist"></a>**Write a 30-word commentary on Finance and Business News.** hopefully related to Finance and Business News.<br/>Southwest Airlines has 3,000 flights canceled during the busy spring break period<br/>Created at 08:13  - [Business] -  generated  by  news  article  robot  (news_art...<br/>Southwest Airlines (NYSE: LUV) has canceled more than 3,000 flights over the busy spring break period amid the carrier's staffing shortage.<br/>created at: 2023-04-19T08:13.159Z |
|                                                              |                                                              |

