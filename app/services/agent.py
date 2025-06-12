from agno.agent import Agent
from agno.models.google import Gemini
from agno.media import Image
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.baidusearch import BaiduSearchTools
# from agno.tools.googlesearch import GoogleSearchTools
from agno.tools.arxiv import ArxivTools
from agno.team.team import Team
import os
from dotenv import load_dotenv


# Import restructured for the agent
from pydantic import BaseModel, Field
from typing import List, Optional

load_dotenv()
# os.environ["GOOGLE_CSE_ID"] = os.getenv("GOOGLE_CSE_ID")
# os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

# Class structured agent
class SkinConcern(BaseModel):
    name: str = Field(..., description="Name of the skin concern")
    severity: str = Field(..., description="Severity level (Mild/Moderate/Severe)")
    type: Optional[str] = Field(None, description="Specific type of concern")
    confidence: float = Field(..., description="Confidence score between 0.0 and 1.0")


class SkincareProducts(BaseModel):
    title: str = Field(..., description="Title of the skincare product")
    description: str = Field(..., description="Description of the skincare product")
    priority: str = Field(..., description="Priority level (High/Medium/Low)")
    # link: str = Field(..., description="Link to the product page")
    price: str = Field(..., description="Price of the product")
    how_to_use: str = Field(..., description="How to use the product")
    benefits: str = Field(..., description="Benefits of the product")
    side_effects: str = Field(..., description="Side effects of the product")
    dosage: str = Field(..., description="Dosage instructions for the product")
    

class Recommendation(BaseModel):
    title: str = Field(..., description="Title of the recommendation")
    description: str = Field(
        ..., description="Detailed description of the recommendation"
    )
    priority: str = Field(..., description="Priority level (High/Medium/Low)")


class AnalysisMetrics(BaseModel):
    skin_hydration: int = Field(
        ..., ge=0, le=100, description="Skin hydration score (0-100)"
    )
    texture_uniformity: int = Field(
        ..., ge=0, le=100, description="Texture uniformity score (0-100)"
    )
    pore_visibility: int = Field(
        ..., ge=0, le=100, description="Pore visibility rating (0-100)"
    )
    overall_score: int = Field(
        ..., ge=0, le=100, description="Overall skin health score (0-100)"
    )


class SkinAnalysisResponse(BaseModel):
    overall_health: str = Field(
        ..., description="General skin health status (Good/Fair/Poor)"
    )
    skin_type: str = Field(..., description="Skin type (Oily/Dry/Combination/Normal)")
    concerns: List[SkinConcern] = Field(
        ..., description="List of identified skin concerns"
    )
    recommendations: List[Recommendation] = Field(
        ..., description="List of treatment recommendations"
    )
    analysis_metrics: AnalysisMetrics = Field(
        ..., description="Quantitative analysis metrics"
    )
    skincare_products: List[SkincareProducts] = Field(
       ..., description="List of recommended skincare products"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "overall_health": "Poor",
                "skin_type": "Oily",
                "concerns": [
                    {
                        "name": "Acne",
                        "severity": "Severe",
                        "type": "Inflammatory",
                        "confidence": 0.95,
                    }
                ],
                "recommendations": [
                    {
                        "title": "Consult a Dermatologist",
                        "description": "Given the severity of the acne, it is highly recommended to consult a dermatologist.",
                        "priority": "High",
                    }
                ],
                "analysis_metrics": {
                    "skin_hydration": 40,
                    "texture_uniformity": 30,
                    "pore_visibility": 85,
                    "overall_score": 35,
                },
                "skincare_products": [
                    {
                    "title": "Acne Cream",
                    "description": "Acne Cream is a gentle and effective treatment for acne. It helps to break down the dead skin cells, reducing the appearance of pores and wrinkles. Acne Cream is available in various formulations, including moisturizing, hydrating, and anti-inflammatory.",
                    "priority": "High",
                    # "link": "URL_ADDRESS.amazon.com/Acne-Cream-100-Count/dp/B07K54K21K",
                    "price": "$20.00",
                    "how_to_use": "Apply the product to the affected area, follow the instructions on the package, and rinse thoroughly after use.",
                    "benefits": "Acne Cream helps break down dead skin cells, reducing the appearance of pores and wrinkles.",
                    "side_effects": "There are no known side effects associated with Acne Cream.",
                    "dosage": "Apply the product according to the instructions on the package.",
                    },
                ]
            }
        }
        





def analyze_skin(image_url, api_key=GEMINI_API_KEY, country=None, journals=None):
    # Set up Agno Agent with Gemini model
    search_agent = Agent(
        name="Searching",
        role="You are a search agent that can search the web for relevant information about the skin problem and how to solve it.",
        model=Gemini(id="gemini-2.0-flash-exp", api_key=api_key),
        tools=[DuckDuckGoTools(), BaiduSearchTools()],
        add_name_to_instructions=True,
        instructions=f"""
        When searching for skin-related information:
        1. Focus on finding reliable medical sources (Mayo Clinic, WebMD, dermatology journals)
        2. Prioritize recent research and studies (last 3 years)
        3. Look for both treatment options and prevention methods
        4. Include information about different skin types and conditions
        5. Verify information from multiple reputable sources
        6. Pay special attention to:
        - Acne and acne scars treatments
        - Hyperpigmentation solutions
        - Anti-aging recommendations
        - Skin hydration techniques
        - Sun protection methods
        7. Always include source links for reference
        8. Present findings in clear, organized bullet points
        9. Give recommendations scincare product that available in {country} and recommended by many people such as writers, or skincare product reviewers or from beauty articles in operating in the {country}. You can use e-commerce products that operate in the {country}.
        """,
    )

    research_agent = Agent(
        name="Researcher",
        role="You are a researcher that can research the research papers for relevant information about the skin problem and how to solve it.",
        model=Gemini(id="gemini-2.0-flash-exp", api_key=api_key),
        tools=[ArxivTools()],
        add_name_to_instructions=True,
        instructions="""
        When researching skin-related information:
        1. Focus exclusively on peer-reviewed dermatology journals and medical research papers
        2. Prioritize studies published within the last 2 years for the most current findings
        3. Search for:
        - Clinical trial results for new treatments
        - Emerging skin conditions and their treatments
        - Breakthrough therapies and their efficacy rates
        - Comparative studies between treatment methods
        4. Always verify findings across multiple reputable sources
        5. Include complete citation information (authors, journal, DOI)
        6. Present findings in this structured format:
        [Condition/Problem]
        - Latest research findings
        - Treatment options (with success rates)
        - Potential side effects
        - Recommended protocols
        7. Highlight any FDA-approved treatments separately
        8. Include links to full papers when available
        9. For controversial topics, present both sides with evidence
        """,
    )

    image_agent = Agent(
        model=Gemini(id="gemini-2.0-flash-exp", api_key=api_key),
        agent_id="dermatologist",
        name="Skin Dermatologist",
        markdown=True,
        instructions=[
            "You are a dermatologist AI that analyzes skin conditions from images.",
            "Compare the original image with the predicted image. Take the prediction point that you think is correct based on the original image.",
            "Provide detailed descriptions and potential diagnoses based on the images you receive (original image).",
        ],
    )

    
    if journals is not None or journals != []:
        agent = Team(
            name="Skin Dermatologist Team",
            mode="route",
            model=Gemini(
                id="gemini-2.0-flash-exp", api_key=api_key),  # Using the strongest multi-modal model
            members=[image_agent, search_agent, research_agent],
            instructions=f"""
            As a dermatologist expert team, your responsibilities are:
            1. Analyze skin images with clinical precision
            2. Analyze skin journals user for accurate reccomendations treatment, {journals}
            3. Collaborate with search and research agents to:
            - Verify diagnosis with latest medical information
            - Cross-reference treatment options
            - Identify emerging therapies
            4. Provide comprehensive analysis including:
            - Condition identification
            - Severity assessment
            - Root cause analysis
            5. Create personalized treatment plans that consider:
            - Skin type
            - Medical history
            - Lifestyle factors
            - Budget considerations
            6. Present information in clear, patient-friendly language
            7. Always include:
            - Primary recommended treatment
            - Alternative options
            - Prevention strategies
            - Expected timeline for results
            8. For complex cases:
            - Consult with research agent for latest studies
            - Verify with search agent for clinical guidelines
            - Present multiple approaches with pros/cons
            9. Maintain professional medical standards in all recommendations
            """,
            show_tool_calls=True,
            markdown=True,
            debug_mode=True,
            show_members_responses=True,
            enable_team_history=True,
            use_json_mode=True,
            response_model=SkinAnalysisResponse,
        )
    else:
        agent = Team(
            name="Skin Dermatologist Team",
            mode="route",
            model=Gemini(
                id="gemini-2.0-flash-exp", api_key=api_key),  # Using the strongest multi-modal model
            members=[image_agent, search_agent, research_agent],
            instructions=f"""
            As a dermatologist expert team, your responsibilities are:
            1. Analyze skin images with clinical precision
            2. Collaborate with search and research agents to:
            - Verify diagnosis with latest medical information
            - Cross-reference treatment options
            - Identify emerging therapies
            3. Provide comprehensive analysis including:
            - Condition identification
            - Severity assessment
            - Root cause analysis
            4. Create personalized treatment plans that consider:
            - Skin type
            - Medical history
            - Lifestyle factors
            - Budget considerations
            5. Present information in clear, patient-friendly language
            6. Always include:
            - Primary recommended treatment
            - Alternative options
            - Prevention strategies
            - Expected timeline for results
            7. For complex cases:
            - Consult with research agent for latest studies
            - Verify with search agent for clinical guidelines
            - Present multiple approaches with pros/cons
            8. Maintain professional medical standards in all recommendations
            """,
            show_tool_calls=True,
            markdown=True,
            debug_mode=True,
            show_members_responses=True,
            enable_team_history=True,
            use_json_mode=True,
            response_model=SkinAnalysisResponse,
        )

    # Analyze the skin image
    analysis_prompt =f"""
    Analyze this facial skin image in detail and provide a structured assessment in the following format:

    1. Overall Skin Health Assessment:
    - Evaluate the general condition of the skin
    - Determine the skin type (Oily, Dry, Combination, Normal)
    - Provide an overall health rating

    2. Identify and Analyze Specific Concerns:
    - List all visible skin concerns
    - For each concern, specify:
        * Name of the condition
        * Severity level (Mild, Moderate, Severe)
        * Specific type if applicable
        * Confidence level of detection (0.0 to 1.0)

    3. Treatment Recommendations:
    - Provide specific, actionable recommendations
    - For each recommendation:
        * Clear title
        * Detailed description
        * Priority level (High, Medium, Low)

    4. Quantitative Metrics:
    - Skin hydration level (0-100)
    - Texture uniformity score (0-100)
    - Pore visibility rating (0-100)
    - Overall skin health score (0-100)
    
    5. Skincare Product Recommendations:
    - List recommended skincare products
    - For each product:
        * Title
        * Description
        * Priority level (High, Medium, Low)
        * Link to the product page
        * Price with currency symbols that have been adjusted to the country of {country}
        * How to use
        * Benefits
        * Side effects
        * Dosage instructions

    Format your response as a Python dictionary exactly matching this structure:
     1. Required Fields:
    - overall_health: A string indicating general skin health status (e.g., "Good", "Fair", "Poor")
    - skin_type: A string specifying skin type (e.g., "Oily", "Dry", "Combination", "Normal")
    
    2. concerns: A JSON array of objects, each containing:
    {{
        "name": "Name of the skin concern",
        "severity": "Severity level (Mild/Moderate/Severe)",
        "type": "Specific type of concern or null",
        "confidence": "Confidence score between 0.0 and 1.0"
    }}
    
    3. recommendations: A JSON array of objects, each containing:
    {{
        "title": "Title of the recommendation",
        "description": "Detailed description of the recommendation",
        "priority": "Priority level (High/Medium/Low)"
    }}
    
    4. analysis_metrics: A JSON object containing:
    {{
        "skin_hydration": "Score from 0-100",
        "texture_uniformity": "Score from 0-100",
        "pore_visibility": "Score from 0-100",
        "overall_score": "Score from 0-100"
    }}
    
    5. skincare_products: A JSON array of objects, each containing:
    {{
        "title": "Title of the skincare product",
        "description": "Description of the skincare product",
        "priority": "Priority level (High/Medium/Low)",
        "price": "Price of the product",
        "how_to_use": "How to use the product",
        "benefits": "Benefits of the product",
        "side_effects": "Side effects of the product",
        "dosage": "Dosage instructions for the product"
    }}

    Ensure all fields are present and properly formatted as they are required by the database schema.
    Return the analysis in a format that exactly matches these database fields.
    """

    response = agent.run(analysis_prompt, images=[Image(filepath=image_url)])
    response_json = response.content.model_dump_json(indent=2)

    return response_json