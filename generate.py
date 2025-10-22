"""Generate SQL queries using Gemini LLM with schema/context examples."""

import os
from typing import List, Dict

from dotenv import load_dotenv
from google import genai
from google.genai import types

from validate import validate_sql

load_dotenv()


class SQLGenerator:
    """SQL Generator using Gemini LLM and validation."""

    def __init__(self, model_name: str = "models/gemini-2.5-pro"):
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "GEMINI_API_KEY not set in environment"
            )

        # Initialize Gemini client
        self.client = genai.Client(api_key=self.api_key)
        self.model_name = model_name

    def _build_prompt(
        self,
        user_question: str,
        contexts: List[Dict],
        max_context_chars: int = 3000
    ) -> str:

        """Build the prompt combining context schemas and examples."""
        lines = [
                "You are a SQL expert. Given these schemas and examples, "
                "generate a single SQL SELECT statement. Output only the SQL "
                "query and nothing else.\n"
        ]
        used = 0
        for ctx in contexts:
            text = ctx.get("text", "")
            example = ctx.get("answer", "")
            if used >= max_context_chars:
                break
            remaining = max_context_chars - used
            text_trunc = text[:remaining]
            used += len(text_trunc)
            lines.extend([
                f"Schema: {text_trunc}",
                f"Example: {example}",
                ""
            ])

        lines.append(f"User question: {user_question}")
        lines.append("SQL:")
        return "\n".join(lines)

    def generate_query(self, user_question: str, contexts: List[Dict]) -> Dict:
        """Generate a SQL query for the given user question and contexts."""
        prompt = self._build_prompt(user_question, contexts)

        try:
            # Generate SQL using Gemini
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.2  # optional: adjust for creativity
                )
            )
            sql_text = response.text.strip()
            # Extract only the SQL if Gemini returns markdown/code
            if "```sql" in sql_text.lower():
                sql_text = sql_text.split("```sql")[1].split("```")[0].strip()
            elif "```" in sql_text:
                sql_text = sql_text.split("```")[1].split("```")[0].strip()

        except Exception:  # noqa: W0718
            # Broad exception used because Gemini API can raise errors
            fallback = contexts[0].get("answer", "") if contexts else ""
            if fallback:
                is_valid, message, formatted = validate_sql(fallback)
            else:
                is_valid, message, formatted = False, (
                    "Gemini error. Fallback used."
                ), ""
            return {
                "sql": formatted if is_valid else fallback,
                "valid": is_valid,
                "validation_message": message,
            }

        # Validate the generated SQL
        is_valid, message, formatted = validate_sql(sql_text)
        return {
            "sql": formatted if is_valid else sql_text,
            "valid": is_valid,
            "validation_message": message,
        }
