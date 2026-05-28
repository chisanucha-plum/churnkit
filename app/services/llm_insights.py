"""LLM integration for AI insights."""

import logging
from typing import Dict, Any, Optional

from app.config.settings import settings

logger = logging.getLogger(__name__)


class LLMInsightsService:
    """Generate AI insights using LLM."""

    def __init__(self):
        """Initialize LLM service."""
        self.enabled = settings.LLM_ENABLED
        self.api_key = settings.LLM_API_KEY
        self.model = settings.LLM_MODEL
        self.client = None

        if self.enabled:
            self._initialize_client()

    def _initialize_client(self) -> None:
        """Initialize LLM client."""
        try:
            from openai import OpenAI

            self.client = OpenAI(api_key=self.api_key)
            logger.info(f"LLM client initialized with model: {self.model}")
        except ImportError:
            logger.warning("OpenAI client not installed")
            self.enabled = False

    def generate_insights(
        self, customer_data: Dict[str, Any], prediction: float
    ) -> Optional[str]:
        """Generate AI insights for a customer."""
        if not self.enabled or self.client is None:
            logger.warning("LLM insights not enabled")
            return None

        try:
            prompt = self._build_prompt(customer_data, prediction)
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.7,
            )

            insights = response.choices[0].message.content
            logger.info("Generated LLM insights")
            return insights

        except Exception as e:
            logger.error(f"Error generating LLM insights: {str(e)}")
            return None

    def _build_prompt(
        self, customer_data: Dict[str, Any], prediction: float
    ) -> str:
        """Build prompt for LLM."""
        return f"""
        Analyze this customer data and provide brief retention recommendations:
        
        Customer Data:
        - Tenure: {customer_data.get('tenure')} months
        - Monthly Charges: ${customer_data.get('monthly_charges')}
        - Contract: {customer_data.get('contract_type')}
        - Internet Service: {customer_data.get('internet_service')}
        - Payment Method: {customer_data.get('payment_method')}
        - Churn Risk: {prediction:.1%}
        
        Provide 2-3 specific retention strategies in 2-3 sentences.
        """

    def generate_batch_insights(
        self, predictions: list
    ) -> Dict[str, Optional[str]]:
        """Generate insights for batch predictions."""
        insights = {}

        for pred in predictions:
            customer_id = pred.get("customer_id")
            insights[customer_id] = self.generate_insights(
                pred.get("customer_data"), pred.get("churn_probability")
            )

        return insights
