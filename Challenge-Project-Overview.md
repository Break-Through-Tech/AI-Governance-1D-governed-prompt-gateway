
# Governed Prompt Gateway: Token Efficient and safety aware LLM routing for regulated customer

**Company / Org:** Other  
**Challenge Advisor:** Bhavana Unnam, unnam29@gmail.com    
**Program:** Break Through Tech AI Studio - Fall 2026  
**AI Studio Coach:** Om Kamath, om.kamath@breakthroughtech.org


---

## 🏢 About Other
Other operates within the specialized domain of enterprise AI governance, focusing on the secure and cost-efficient deployment of generative models in high-stakes industries. The team is dedicated to building robust frameworks that bridge the gap between experimental AI capabilities and the rigorous compliance standards required by regulated customer service environments.

---

## 🎯 The Challenge
### Project Summary
In this project, you will use publicly available customer-service, insurance question answering, toxic prompt, jailbreak, and hallucination datasets and NLP classification, semantic similarity, prompt optimization, retrieval-augmented generation, caching, and LLM evaluation techniques to build a governed prompt gateway that screens user queries, detects unsafe requests, reduces token usage, estimates cost, and recommends the best LLM model tier before a prompt is executed. This will help our company address the business problem of deploying generative AI safely and cost effectively by reducing LLM spend, improving response reliability, and creating an auditable governance layer for regulated customer service use cases.

### Success Criteria

**Safety and governance accuracy**

- Target success metric:
  - Achieve strong baseline performance on harmful-query and risk classification.
  - Measure accuracy, precision, recall, and F1 score.
  - A successful target would be 80%+ F1 score on prompt-risk classification.

**Token reduction and cost efficiency**

- Target success metric:
  - Compare original prompt token count against optimized prompt token count.
  - A successful target would be 30–50% reduction in input tokens for eligible prompts.
  - Estimate cost savings across a simulated workload of customer-service queries.

**Cache effectiveness**

- Target success metric:
  - Measure cache-hit rate across a test set with repeated and similar queries.

**A successful December demo:**

- User enters a query.
- System classifies the query intent and risk.
- Unsafe queries are blocked or redirected.
- Safe queries are optimized for token reduction.
- Similar queries trigger cache reuse.
- Dashboard displays token savings, cost estimate, risk score, and recommended model tier.
- Final response is generated or simulated with an auditable decision log.

### Stretch Goals

There are several ways to stretch this project but i fear this is already too much for an implementation for grads. I am open to add one or two scopes though.

- **Jurisdiction-aware policy profiles**
  - Fellows can create configurable policy templates for different regions or industries, such as insurance, banking, healthcare, or education. The goal would not be to provide legal advice, but to show how enterprise AI governance rules can be customized.
- **Automated prompt testing suite**
  - Fellows can build a test harness that runs many prompts through the gateway and reports safety accuracy, token reduction, cache performance, cost estimates, and routing consistency.

### Project Milestones
Use these milestones to guide your work. Your team will create a GitHub Projects board to track tasks within each milestone.
| Month | Milestone | Key Activities |
|---|---|---|
| September | Problem Definition, Data Setup, and Baseline Design | • Finalize project scope and success criteria.<br>• Select public datasets and complete initial data exploration.<br>• Define prompt-risk categories and governance rules.<br>• Build a basic data pipeline for loading and preprocessing text data.<br>• Create a baseline query classifier for intent and risk detection.<br>• Establish baseline metrics for safety classification, token usage, and response quality. |
| October | Governance Layer, Prompt Optimization, and Caching Prototype | • Build the first working version of the prompt governance layer.<br>• Implement harmful-query and jailbreak-detection logic.<br>• Add prompt compression or prompt-refinement functionality.<br>• Implement semantic similarity or embedding-based caching.<br>• Measure token reduction compared to baseline prompts. |
| November | Evaluation Dashboard, and Final Demo Preparation | • Complete the end to end governed prompt gateway workflow.<br>• Add model recommendation logic based on cost, complexity, risk, and use case.<br>• Finalize the dashboard showing governance decisions, token savings, cost estimates, cache status, and model recommendation.<br>• Evaluate the system using metrics such as harmful-query detection accuracy, token reduction, cache-hit rate, and cost-savings estimate.<br>• Prepare final presentation, architecture diagram, demo script, and project documentation. |

> **Note for the team:** Please create a GitHub Projects board in this repository to break these milestones into weekly tasks. Go to the **Projects** tab → **New project** → Choose **Board** → Add columns for each month.

---

## 📊 Dataset
**Name and Source:** [LMSYS Toxic Chat](https://huggingface.co/datasets/lmsys/toxic-chat), [JailbreakBench](https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors)  
**Format:** CSV/TSV  
**Size:** under 1gb  
**Location:** Hosted on HuggingFace Datasets  

### Key Details
- [Brief description of what's in the data]
- [Any known limitations or preprocessing needed]
- [Link to data dictionary or documentation, if available]
  
---

## 🛠️ Suggested Approach
**ML Problem Type:** NLP & RAG  

**Recommended Libraries:**
- [e.g., pandas, scikit-learn, TensorFlow, Hugging Face]
  
**Evaluation Metrics:**
- [e.g., Accuracy, Precision/Recall, RMSE, BLEU score]

---

## 📚 Resources to Get Started

The following resources will help your team understand the problem space and potential technical approaches for this project:

**Background Reading:**
- [e.g., Link to an article or blog post about the problem domain]
- [e.g., Link to an industry report or case study]

**Technical Tutorials:**
- [e.g., Link to a free tutorial on the ML technique(s) involved]
- [e.g., Link to documentation for a key library or tool]

**Code Examples:**
- [e.g., Link to a relevant GitHub repo]
- [e.g., Link to a sample implementation or starter code]

**Other:**
- [Links to any additional resources — e.g., papers, videos, podcasts, etc.]

*Feel free to explore beyond these, and share anything interesting you find with me!*

---

## 🤝 How We'll Work Together

**Official check-ins:** During our biweekly 45-minute AI Studio Lab Section meeting block (2nd and 4th week of every month)

 **Other ways to reach out to me with questions:** 
* [e.g., Your team's channel within Break Through Tech’s Discord space]
* [e.g., Email; please copy your teammates and AI Studio Coach]
* [e.g., Request a team check-in on Zoom]
* [Note: I will aim to respond within 48 hours. Please reach out to your AI Studio Coach with urgent questions.]

> 💡 **Challenge Advisor: Please update the above based on your availability and preference. If you are not able to answer questions or meet with fellows outside of the biweekly Lab Section check-ins, simply write in "N/A (only available during the official check-in times)"**

**Recommended free coding / collaboration tools**
* […]
* […]

---

## 🚀 Getting Started

1. **Review this overview document** and note any questions for our first meeting
2. **Begin reviewing the dataset** using the link above
3. **Read the GitHub Projects documentation** [here](https://docs.github.com/en/issues/planning-and-tracking-with-projects/learning-about-projects/about-projects)

I’m excited to work with you!

---

## ❓ Questions?

Please bring any questions to our first meeting during the week of August 24th (Break Through Tech’s Bridge to Studio - Session C). 
