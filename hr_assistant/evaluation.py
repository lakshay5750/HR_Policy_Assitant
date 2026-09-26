"""
Step 9: evaluate answer quality against a fixed set 
of test questions.

Unlike tracing (which just records what happened), 
evaluation runs the
agent against a known set of question/reference-answer 
pairs and scores
each answer using a second LLM as a judge.
Results are uploaded to
LangSmith as a Dataset + Experiment, 
so quality can be compared across
runs (after a prompt change, a new model, a new guardrail, etc).

The judge model is routed through Portkey too,
using the same slug as
the main app's LLM (gateway.py's PRIMARY_PROVIDER) 
but a different
underlying model (JUDGE_MODEL_NAME) - 
so it isn't grading its own
output verbatim, without needing a second slug set up.
"""


from langchain_openai import ChatOpenAI
from langsmith import Client
from openevals.llm import create_llm_as_judge
from openevals.prompts import CORRECTNESS_PROMPT, RAG_GROUNDEDNESS_PROMPT
from portkey_ai import createHeaders, PORTKEY_GATEWAY_URL

from hr_assistant import config
from hr_assistant.gateway import PRIMARY_PROVIDER
from hr_assistant.logger import get_logger
from hr_assistant.pipeline import ask, build_hr_assistant
from hr_assistant.vector_store import get_retriever, load_vector_store



logger = get_logger(__name__)

# question paper 
DATASET_NAME = "hr-policy-qna"

TEST_CASES=[
  {
    "question": "What is Microsoft's mission?",
    "answer": "To empower every person and every organization on the planet to achieve more."
  },
  {
    "question": "What values are emphasized in Microsoft's Standards of Business Conduct?",
    "answer": "Respect, integrity, and accountability."
  },
  {
    "question": "What areas do Microsoft's employee benefits support?",
    "answer": "Physical wellbeing, mental wellbeing, emotional wellbeing, financial wellbeing, time away from work, family support, flexible work, learning and education, career development, employee resource groups, workplace perks, performance-based compensation, bonuses, and stock awards."
  },
  {
    "question": "Do Microsoft benefits vary by employee?",
    "answer": "Yes. Benefits vary according to employee location, role, and applicable benefits program."
  },
  {
    "question": "Does Microsoft have one universal leave entitlement for all employees worldwide?",
    "answer": "No. Microsoft does not have one publicly stated universal leave entitlement for every employee worldwide."
  },
  {
    "question": "What types of time-away programs does Microsoft publicly describe?",
    "answer": "Vacation, sick time, holidays, parental leave, family caregiver leave, military leave, bereavement leave, jury duty, and unpaid personal leave."
  },
  {
    "question": "What determines an employee's leave eligibility and entitlement?",
    "answer": "Employee classification, location, and applicable benefits plans."
  },
  {
    "question": "How much parental leave can eligible U.S. Microsoft employees receive?",
    "answer": "Eligible U.S. employees can take up to 12 weeks of parental leave."
  },
  {
    "question": "How much additional maternity disability leave may eligible U.S. birth mothers or persons receive?",
    "answer": "They may be eligible for an additional eight weeks of maternity disability leave."
  },
  {
    "question": "How much total leave can an eligible U.S. birth mother or person potentially receive?",
    "answer": "Up to 20 weeks, including up to 12 weeks of parental leave and an additional eight weeks of maternity disability leave."
  },
  {
    "question": "How much family caregiver leave can eligible U.S. Microsoft employees receive?",
    "answer": "Up to 12 weeks."
  },
  {
    "question": "How is U.S. family caregiver leave paid?",
    "answer": "The first 4 weeks are paid at 100% of salary and the remaining 8 weeks are unpaid."
  },
  {
    "question": "How long can short-term disability leave last for eligible U.S. Microsoft employees?",
    "answer": "Up to 26 weeks."
  },
  {
    "question": "How is U.S. short-term disability leave paid?",
    "answer": "The first 8 weeks are paid at 100% of salary and the remaining weeks are paid at 75% of salary."
  },
  {
    "question": "How much military leave pay can eligible U.S. Microsoft employees receive?",
    "answer": "Up to 30 days of full pay without offset per calendar year for military duty."
  },
  {
    "question": "Does Microsoft have vacation and Discretionary Time Off programs?",
    "answer": "Yes. Microsoft's U.S. benefits information describes vacation and Discretionary Time Off programs."
  },
  {
    "question": "Does Microsoft provide floating holidays?",
    "answer": "Yes. Microsoft publicly describes floating holidays for eligible employees."
  },
  {
    "question": "How many floating holidays may eligible U.S. Microsoft employees receive per year?",
    "answer": "Eligible employees may receive four floating holidays per year."
  },
  {
    "question": "Are part-time employee benefits prorated?",
    "answer": "Part-time employee benefits may be prorated according to applicable rules."
  },
  {
    "question": "What types of workplace arrangements does Microsoft support?",
    "answer": "Remote, hybrid, and fully on-site arrangements."
  },
  {
    "question": "What factors can determine Microsoft's work arrangement for an employee?",
    "answer": "Role, team, location, business requirements, customer needs, and collaboration requirements."
  },
  {
    "question": "Does Microsoft have one remote-work rule for every employee?",
    "answer": "No. Employees should not assume that one remote-work rule applies to every Microsoft employee."
  },
  {
    "question": "Where can an employee find the expected worksite arrangement?",
    "answer": "Individual Microsoft job postings can specify the expected worksite arrangement."
  },
  {
    "question": "What factors does Microsoft consider when determining compensation?",
    "answer": "Role, level, location, current market data, and individual contribution."
  },
  {
    "question": "What can Microsoft's compensation include?",
    "answer": "Base pay, bonuses, stock awards, and other benefits."
  },
  {
    "question": "What factors can Microsoft consider for bonus-eligible roles?",
    "answer": "Both what an employee achieves and how the employee achieves it."
  },
  {
    "question": "Can Microsoft roles include stock awards?",
    "answer": "Yes. Many roles can include stock awards that vest over time."
  },
  {
    "question": "Does compensation vary by country at Microsoft?",
    "answer": "Yes. Compensation and benefits vary by country."
  },
  {
    "question": "What does Microsoft consider when discussing pay equity?",
    "answer": "Employees performing substantially similar work should be compensated fairly while considering legitimate factors such as role, level, and tenure."
  },
  {
    "question": "Does Microsoft state that it provides equal employment opportunity?",
    "answer": "Yes. Microsoft states that it does not discriminate against employees or applicants based on protected characteristics."
  },
  {
    "question": "What employment activities are covered by Microsoft's equal employment opportunity policy?",
    "answer": "Recruitment, hiring, training, compensation, promotion, benefits, social and recreational programs, and discipline."
  },
  {
    "question": "Does Microsoft provide reasonable accommodation?",
    "answer": "Yes. Microsoft states that it provides reasonable accommodation to qualified employees with protected disabilities to the extent required by applicable laws and regulations where they work."
  },
  {
    "question": "What can Microsoft's reasonable accommodation process depend on?",
    "answer": "Applicable local law and the employee's circumstances."
  },
  {
    "question": "What topics are covered by Microsoft's Trust Code?",
    "answer": "Employee responsibilities, manager responsibilities, ethical decision-making, business conduct, diversity and inclusion, conflicts of interest, security, confidential information, business integrity, and building trust."
  },
  {
    "question": "What should Microsoft employees do about potential conflicts of interest?",
    "answer": "They should act in Microsoft's best interests, avoid situations where personal relationships or financial interests could improperly influence business decisions, disclose potential conflicts, and seek appropriate advice or approval."
  },
  {
    "question": "Can outside activities interfere with Microsoft's interests?",
    "answer": "Outside activities should not interfere with Microsoft's interests or violate applicable agreements and policies."
  },
  {
    "question": "How can Microsoft employees report compliance concerns?",
    "answer": "Employees can use Microsoft's Integrity Portal and other designated reporting channels."
  },
  {
    "question": "What types of concerns does Microsoft encourage employees to report?",
    "answer": "Potential violations of the Standards of Business Conduct, Microsoft policies, applicable law, or compliance requirements."
  },
  {
    "question": "Does Microsoft prohibit retaliation for raising compliance concerns?",
    "answer": "Microsoft states that retaliation against someone for raising a compliance concern can result in disciplinary action."
  },
  {
    "question": "What are the three major areas of Microsoft's compliance and ethics program?",
    "answer": "Prevention, detection, and remediation."
  },
  {
    "question": "What activities can be part of Microsoft's compliance prevention program?",
    "answer": "Standards of Business Conduct, policies, training, risk assessment, data analytics, and third-party vetting."
  },
  {
    "question": "What activities can be part of Microsoft's compliance detection program?",
    "answer": "Internal audit testing, compliance analytics, controls, investigations, and trend analysis."
  },
  {
    "question": "What does remediation focus on at Microsoft?",
    "answer": "Identifying root causes and improving controls and processes."
  },
  {
    "question": "What can happen if an employee violates Microsoft's policies and standards?",
    "answer": "The employee can face disciplinary action, potentially including termination of employment."
  },
  {
    "question": "What conduct does Microsoft's anti-corruption program prohibit?",
    "answer": "Bribery, kickbacks, improper payments, corrupt benefits, and improper attempts to obtain or retain business."
  },
  {
    "question": "What types of employee training does Microsoft publicly describe?",
    "answer": "Initial training, annual refresher training, role-based training where applicable, and Standards of Business Conduct training."
  },
  {
    "question": "What can employee training requirements depend on?",
    "answer": "The employee's role and responsibilities."
  },
  {
    "question": "What happens to employee access when employment ends?",
    "answer": "Access to company systems and resources can be removed according to applicable processes."
  },
  {
    "question": "Why does Microsoft remove access when employment ends?",
    "answer": "To help prevent former employees from retaining unauthorized access."
  },
  {
    "question": "Does the Microsoft public HR dataset provide an exact notice period for every employee?",
    "answer": "No. The dataset states that an exact notice period for every Microsoft employee should not be invented unless an official source is available."
  },
  {
    "question": "Does the Microsoft public HR dataset provide exact annual leave entitlement for every country?",
    "answer": "No. The dataset does not provide a universal exact annual leave entitlement for every country."
  },
  {
    "question": "Does the Microsoft public HR dataset provide exact employee salaries?",
    "answer": "No. Exact employee salaries should not be invented from the available knowledge base."
  },
  {
    "question": "Does the Microsoft public HR dataset contain individual employee compensation information?",
    "answer": "No. Individual employee compensation information should not be invented or provided as if it were in the knowledge base."
  },
  {
    "question": "Does the Microsoft public HR dataset contain internal HR contact details?",
    "answer": "No. Internal HR contact details are not available in the public knowledge base."
  },
  {
    "question": "What should the assistant say when an HR policy is unavailable in the knowledge base?",
    "answer": "I couldn't find that information in the available public Microsoft HR knowledge base. I don't want to invent a policy. Please consult the applicable Microsoft HR/benefits resource."
  },
  {
    "question": "Can U.S.-specific Microsoft benefits automatically be applied to employees in other countries?",
    "answer": "No. U.S.-specific benefits must not automatically be applied to employees in another country."
  },
  {
    "question": "Can the U.S. 12-week parental leave rule automatically be used to answer questions about Microsoft employees in India?",
    "answer": "No. The available public information contains U.S.-specific parental leave information and does not establish a universal entitlement for Microsoft employees in India."
  },
  {
    "question": "What should the RAG assistant do if the retrieved documents do not contain an answer?",
    "answer": "It should not guess, should not use a generic HR policy as Microsoft's policy, should not create an exact number, and should state that the information is unavailable."
  }
]

JUDGE_MODEL_NAME = "openai/gpt-oss-20b"


# making  a judge llm

def _get_judge_llm() -> ChatOpenAI:
    """Return a judge model routed through Portkey, same slug as the main app."""
    headers = createHeaders(api_key=config.PORTKEY_API_KEY, 
            provider=PRIMARY_PROVIDER)
    return ChatOpenAI(api_key="portkey", 
        base_url=PORTKEY_GATEWAY_URL, 
        default_headers=headers, 
        model=JUDGE_MODEL_NAME)
    
# if dataset is there reuse it , if not create a new dataset 
# question paper 
def _ensure_dataset(client: Client):
    """Create the LangSmith dataset if it doesn't exist yet, and upload the test cases."""
    if client.has_dataset(dataset_name=DATASET_NAME):
        logger.info("Dataset '%s' already exists, reusing it", DATASET_NAME)
        return client.read_dataset(dataset_name=DATASET_NAME)

    logger.info("Creating dataset '%s' with %d example(s)", 
        DATASET_NAME, len(TEST_CASES))
    dataset = client.create_dataset(dataset_name=DATASET_NAME)
    client.create_examples(
        dataset_id=dataset.id,
        examples=[
            {"inputs": {"question": case["question"]},
            "outputs": {"answer": case["answer"]}}
            for case in TEST_CASES
        ],
    )
    return dataset

# start the exam
def run_evaluation():
    """Upload the dataset (if needed) 
    and run the correctness evaluation."""
    client = Client()
    dataset = _ensure_dataset(client)

    # Built once and reused for every test case, instead of rebuilding
    # the whole agent (and reconnecting to Qdrant) 10 times over.
    agent = build_hr_assistant()
    retriever = get_retriever(load_vector_store())

    # write the answers 
    def target(inputs: dict) -> dict:
        """
        Run one test question through the real agent, 
        and also capture
        the retrieved chunks 
        so groundedness can check the answer against
        what was actually retrieved 
        (not just the reference answer).
        """
        answer = ask(agent, inputs["question"])
        chunks = retriever.invoke(inputs["question"])
        context = "\n\n".join(chunk.page_content for chunk in chunks)
        return {"answer": answer, "context": context}

    # giving marks 
    correctness_evaluator = create_llm_as_judge(
        prompt=CORRECTNESS_PROMPT,
        feedback_key="correctness",
        judge=_get_judge_llm(),
    )

    groundedness_judge = create_llm_as_judge(
        prompt=RAG_GROUNDEDNESS_PROMPT,
        feedback_key="groundedness",
        judge=_get_judge_llm(),
    )

    def groundedness_evaluator(outputs: dict, **kwargs) -> dict:
        """Check the answer is supported by the retrieved context, not invented."""
        return groundedness_judge(outputs={"answer": outputs["answer"]}, context=outputs["context"])

    logger.info("Running evaluation against dataset '%s'", DATASET_NAME)
    return client.evaluate(
        target,
        data=dataset.name,
        evaluators=[correctness_evaluator,
                groundedness_evaluator],
        experiment_prefix="hr-policy-evalzz",
        description="HR policy assistant correctness + groundedness evaluation",
    )