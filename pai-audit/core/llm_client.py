"""
LLM Integration Module for PAI
Provides AI-powered donation advice using various LLM backends
"""

import os
import json
import streamlit as st
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum


class LLMProvider(Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"  # Ollama or similar
    DEMO = "demo"   # Fallback demo mode


@dataclass
class LLMConfig:
    """LLM configuration."""
    provider: LLMProvider = LLMProvider.DEMO
    model: str = "gpt-4o-mini"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 2000


class LLMDonationAdvisor:
    """
    AI-powered donation advisor using LLM.
    
    Features:
    - Personalized donation strategy based on donor profile
    - Tax optimization advice
    - Charity matching and recommendations
    - Warm-glow effect amplification
    """
    
    SYSTEM_PROMPT = """You are PAI (Philanthropic Asset Intelligence), an AI donation advisor 
    helping donors maximize their charitable impact. You specialize in:
    
    1. **Donation Strategy**: Personalized giving strategies based on donor profile
    2. **Tax Optimization**: IRS-compliant advice on DAFs, bunching, appreciated securities
    3. **Charity Evaluation**: Evidence-based charity recommendations using GiveWell, 
       ACE, and other charity evaluators
    4. **Behavioral Insights**: Applying warm-glow theory (Andreoni 1990) and nudge theory
       (Thaler & Sunstein) to increase donor satisfaction and retention
    
    Important guidelines:
    - Always cite evidence for charity recommendations
    - Be transparent about uncertainty in cost-effectiveness estimates
    - Recommend GiveWell top charities for maximum impact
    - Suggest practical next steps
    - Keep responses concise but informative (max 500 words)
    
    Do NOT:
    - Provide specific legal/tax advice (recommend consulting professionals)
    - Overstate certainty of predictions
    - Recommend risky investments
    """
    
    def __init__(self, config: Optional[LLMConfig] = None):
        """
        Initialize LLM advisor.
        
        Args:
            config: LLM configuration (uses env vars or defaults to demo mode)
        """
        self.config = config or self._load_config()
        self._client = None
        
    def _load_config(self) -> LLMConfig:
        """Load configuration from environment or defaults."""
        # Check for API keys
        api_key = os.getenv('OPENAI_API_KEY') or os.getenv('ANTHROPIC_API_KEY')
        
        if os.getenv('OPENAI_API_KEY'):
            return LLMConfig(
                provider=LLMProvider.OPENAI,
                model=os.getenv('OPENAI_MODEL', 'gpt-4o-mini')
            )
        elif os.getenv('ANTHROPIC_API_KEY'):
            return LLMConfig(
                provider=LLMProvider.ANTHROPIC,
                model=os.getenv('ANTHROPIC_MODEL', 'claude-3-haiku-20240307')
            )
        else:
            return LLMConfig(provider=LLMProvider.DEMO)
    
    def _init_client(self):
        """Initialize the LLM client based on provider."""
        if self._client is not None:
            return
        
        if self.config.provider == LLMProvider.OPENAI:
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    api_key=os.getenv('OPENAI_API_KEY'),
                    base_url=os.getenv('OPENAI_BASE_URL')
                )
            except ImportError:
                print("OpenAI package not installed. Using demo mode.")
                self.config.provider = LLMProvider.DEMO
                
        elif self.config.provider == LLMProvider.ANTHROPIC:
            try:
                from anthropic import Anthropic
                self._client = Anthropic(
                    api_key=os.getenv('ANTHROPIC_API_KEY')
                )
            except ImportError:
                print("Anthropic package not installed. Using demo mode.")
                self.config.provider = LLMProvider.DEMO
    
    def generate_advice(
        self,
        donor_type: str,
        annual_budget: str,
        interest_area: str,
        tax_situation: str,
        charity_data: List[Dict],
        rag_context: str = "",
    ) -> str:
        """
        Generate personalized donation advice.

        Args:
            donor_type: Type of donor (individual, DAF holder, corporate, foundation)
            annual_budget: Annual giving budget
            interest_area: Area of interest (global health, rare disease, education, etc.)
            tax_situation: Tax deduction situation
            charity_data: List of charity information dictionaries
            rag_context: Optional context from Federated RAG retrieval

        Returns:
            Personalized donation advice as markdown string
        """
        if self.config.provider == LLMProvider.DEMO:
            return self._generate_demo_advice(
                donor_type, annual_budget, interest_area, tax_situation, charity_data
            )

        self._init_client()

        # Build the prompt
        user_prompt = self._build_prompt(
            donor_type, annual_budget, interest_area, tax_situation, charity_data
        )

        # Inject RAG context if available
        if rag_context:
            user_prompt = f"## Knowledge Base Context\n{rag_context}\n\n---\n\n{user_prompt}"

        try:
            if self.config.provider == LLMProvider.OPENAI:
                response = self._client.chat.completions.create(
                    model=self.config.model,
                    messages=[
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens
                )
                return response.choices[0].message.content

            elif self.config.provider == LLMProvider.ANTHROPIC:
                response = self._client.messages.create(
                    model=self.config.model,
                    max_tokens=self.config.max_tokens,
                    system=self.SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_prompt}]
                )
                return response.content[0].text

        except Exception as e:
            print(f"LLM API error: {e}. Falling back to demo mode.")
            return self._generate_demo_advice(
                donor_type, annual_budget, interest_area, tax_situation, charity_data
            )
    
    def _build_prompt(
        self,
        donor_type: str,
        annual_budget: str,
        interest_area: str,
        tax_situation: str,
        charity_data: List[Dict]
    ) -> str:
        """Build the user prompt for the LLM."""
        charities_md = "\n".join([
            f"- **{c['name']}**: {c.get('category', 'N/A')}, "
            f"${c.get('cost_per_life', 'N/A')}/life saved, "
            f"Evidence: {c.get('evidence_strength', 'N/A')}, "
            f"Region: {c.get('region', 'N/A')}"
            for c in charity_data[:10]  # Top 10
        ])
        
        return f"""## Donor Profile
- **Donor Type**: {donor_type}
- **Annual Giving Budget**: {annual_budget}
- **Interest Area**: {interest_area}
- **Tax Situation**: {tax_situation}

## Available Charities
{charities_md}

## Task
Provide personalized donation advice including:
1. Top 3 charity recommendations with rationale
2. Tax optimization strategy specific to their situation
3. Practical next steps
4. Expected impact based on their budget

Format response as markdown. Be specific and actionable."""
    
    def _generate_demo_advice(
        self,
        donor_type: str,
        annual_budget: str,
        interest_area: str,
        tax_situation: str,
        charity_data: List[Dict]
    ) -> str:
        """
        Generate demo advice without LLM API.
        
        Uses rule-based logic to simulate personalized advice.
        Note: This is a fallback for demo purposes.
        """
        import random
        
        # Budget mapping
        budget_map = {
            "Under $1,000": 500, 
            "$1,000-$10,000": 5000, 
            "$10,000-$100,000": 50000, 
            "$100,000-$1M": 500000, 
            "Over $1M": 5000000
        }
        budget_num = budget_map.get(annual_budget, 5000)
        
        # Filter charities by interest
        if interest_area == "Rare Disease":
            filtered = [c for c in charity_data if "Rare" in c.get("category", "")]
        else:
            filtered = sorted(charity_data, key=lambda x: x.get("impact_score", 0), reverse=True)[:3]
        
        if not filtered:
            filtered = charity_data[:3]
        
        # Generate recommendations
        recommendations = []
        for i, charity in enumerate(filtered, 1):
            cost = charity.get("cost_per_life", 5000)
            lives = budget_num / cost if cost > 0 else 0
            recommendations.append(f"""
| **{charity['name']}** | |
|------|------|
| Category | {charity.get('category', 'N/A')} |
| Impact Score | ⭐ {charity.get('impact_score', 0):.0%} |
| Cost per Life | ${cost:,} |
| Your Budget Impact | ~{lives:.1f} lives |
| Evidence Strength | {charity.get('evidence_strength', 0):.0%} |
| Region | {charity.get('region', 'N/A')} |
""")
        
        # Tax advice
        tax_advice = {
            "Appreciated Securities": """
- ✅ **Donate appreciated securities** → Avoid capital gains + full deduction
- ✅ **Bunching strategy**: Concentrate 2-3 years of giving into one year
- ✅ **DAF contribution**: Upfront tax benefit + tax-free growth
- 📊 *Evidence: Saves ~20-30% in taxes for high earners*
""",
            "DAF Already Open": """
- ✅ **Optimize DAF investments**: Default allocation likely underperforms 8-12%/year
- ✅ **Increase payout rate**: Consider 5-7% annual grant for more impact
- ✅ **Bunching into DAF**: Maximize tax efficiency
- 📊 *Expected extra annual grants: 2-5% of DAF balance*
""",
            "Itemized Deduction": """
- ✅ **Bunching strategy**: Cluster donations to exceed standard deduction
- ✅ **Appreciated securities > cash**: Save 15-20% in taxes
- ✅ **DAF for long-term giving**: Tax-free growth + future flexibility
- 📊 *Typical tax savings: $2,000-10,000/year for $50K donors*
""",
            "Standard Deduction": """
- ⚠️ **Consider bunching**: Save 2-3 years of donations for one big deduction
- ✅ **DAF opening**: Establish now for future tax benefits
- ✅ **Check state charitable deductions**: May differ from federal
- 📊 *Still worth giving - warm-glow benefits are immediate*
"""
        }
        
        tax_text = tax_advice.get(tax_situation, tax_advice["Standard Deduction"])
        
        # Behavioral insights
        behavioral = f"""
#### 🧠 Behavioral Economics Insights

Based on research:
- **Warm-Glow Effect** (Andreoni 1990): You receive psychological benefits immediately from giving
- **AI Personalization Boost** (White et al. 2026): LLM dialogue increases effective giving by **45.9%**
- **Retention**: First-time donors: 19.4% return → Set up monthly giving ($50-100/mo) → 60%+ retention
- **Commitment Devices**: Make a public pledge → 30% higher follow-through rate

#### 📋 Next Steps

1. **Today**: Donate ${min(budget_num * 0.3, 5000):,.0f} to top recommended charity
2. **This Week**: Research DAF options (Fidelity Charitable, Schwab Charitable, Vanguard Charitable)
3. **This Month**: Set up monthly recurring donation for ongoing impact
4. **Tax Season**: Consult tax advisor on bunching strategy

---
*⚠️ Disclaimer: This is demo mode advice. For actual decisions, please consult financial/tax professionals.*

*🤖 In production, this would use GPT-4o or Claude for personalized advice.*
"""
        
        return f"""### 📋 PAI Personalized Donation Advice

**Donor Profile:** {donor_type} | Budget: {annual_budget} | Interest: {interest_area}

---

#### 🎯 Top Charity Recommendations

{''.join(recommendations)}

---

#### 💰 Tax Optimization Strategy

{tax_text}

---

{behavioral}"""


# Singleton instance for Streamlit
@st.cache_resource
def get_llm_advisor() -> LLMDonationAdvisor:
    """Get cached LLM advisor instance."""
    return LLMDonationAdvisor()


def check_llm_status() -> Dict[str, Any]:
    """
    Check LLM configuration status.
    
    Returns:
        Dict with provider info and availability
    """
    advisor = LLMDonationAdvisor()
    
    status = {
        "provider": advisor.config.provider.value,
        "model": advisor.config.model,
        "available": advisor.config.provider != LLMProvider.DEMO,
        "env_vars": {
            "OPENAI_API_KEY": bool(os.getenv('OPENAI_API_KEY')),
            "ANTHROPIC_API_KEY": bool(os.getenv('ANTHROPIC_API_KEY')),
        },
        "message": ""
    }
    
    if status["available"]:
        status["message"] = f"Using {status['provider']} ({status['model']})"
    else:
        status["message"] = """
🔧 **LLM Configuration Required for Full AI Features**

Set one of the following environment variables:

**Option 1: OpenAI (Recommended)**
```bash
export OPENAI_API_KEY=sk-...
```

**Option 2: Anthropic Claude**
```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

**Option 3: Local (Ollama)**
```bash
export OPENAI_BASE_URL=http://localhost:11434/v1
export OPENAI_API_KEY=ollama
```

Currently running in **demo mode** with rule-based advice.
"""
    
    return status
