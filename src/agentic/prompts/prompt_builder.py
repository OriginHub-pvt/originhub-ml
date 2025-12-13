"""
PromptBuilder
=============

Central place to construct prompts for all agents.

This keeps prompt templates in one location so they are easy
to iterate on and easy to test.
"""

from typing import Any, Dict, List, Optional


class PromptBuilder:
    """
    Builds prompts for each agent in the pipeline.
    """

    def __init__(
        self,
        language: str = "en",
    ):
        """
        Parameters
        ----------
        language : str
            Reserved for future multilingual support. Currently unused.
        """
        self.language = language

    # ------------------------------------------------------------------
    # 1) Interpreter
    # ------------------------------------------------------------------

    def interpreter_prompt(self, input_text: str) -> str:
        """
        Parameters
        ----------
        input_text : str
            Raw user idea text.

        Returns
        -------
        str
            Prompt instructing the model to return a single JSON object.
        """
        return f"""You are an expert product strategist specializing in analyzing and structuring startup ideas. Your task is to parse the user's idea and convert it into a well-organized JSON object.

            User's Idea:
            {input_text}

            CRITICAL INSTRUCTIONS:
            1. Output ONLY a valid JSON object - no explanations, code fences, markdown, or any text before/after JSON
            2. All JSON keys and values must be lowercase strings (except for proper nouns and acronyms)
            3. Keep all text concise and specific

            REQUIRED FIELDS (must be filled):
            - "title": Catchy, clear product name (3-6 words maximum). Example: "Smart Grocery List"
            - "description": 1-2 sentences explaining what the product is and what it does
            - "problem": The specific user problem this idea addresses in 1-2 sentences

            OPTIONAL FIELDS (fill if mentioned):
            - "one_line": Ultra-concise elevator pitch (one sentence, max 15 words)
            - "solution": How the product solves the problem (1-2 sentences)
            - "domain": High-level industry/category (e.g., "productivity", "healthcare", "fintech")
            - "target_users": Who will use this product (be specific: "busy professionals aged 25-45", not "everyone")
            - "key_features": List of 3-7 core features or capabilities
            - "stage": Current development stage - must be one of: "idea", "prototype", "mvp", "launched"
            - "market_size": Estimated market size or target audience size (if mentioned)
            - "differentiator": What makes this unique compared to existing solutions
            - "revenue_model": How the product will make money (e.g., "subscription", "freemium", "b2b")
            - "extra_notes": Any other relevant details not captured above

            QUALITY RULES:
            - Be concise: no fluff or marketing jargon
            - Be specific: avoid vague terms like "innovative" or "powerful"
            - Be accurate: extract only what the user mentioned, don't invent details
            - Use clear language: short sentences, simple words

            EXAMPLE OUTPUT:
            {{"title": "smart meal planner", "description": "An app that suggests weekly meal plans based on dietary preferences and available ingredients.", "problem": "People spend too much time deciding what to eat and planning grocery shopping.", "domain": "health & wellness", "target_users": "busy professionals and families with young children", "key_features": ["personalized meal suggestions", "grocery list sync", "nutrition tracking", "recipe recommendations"], "stage": "idea"}}

            Now parse the user's idea and respond with ONLY the JSON object."""

    # ------------------------------------------------------------------
    # 2) Clarifier
    # ------------------------------------------------------------------

    def clarifier_prompt(self, interpreted: Dict[str, Any]) -> str:
        """
        Parameters
        ----------
        interpreted : dict
            Parsed idea JSON from InterpreterAgent.

        Returns
        -------
        str
            Prompt asking the model to generate field-specific clarifying questions.
        """
        required_fields = [
            ("title", "What is the title of your idea?"),
            ("description", "Please provide a description for your idea."),
            ("problem", "What problem does your idea solve?")
        ]
        missing = [q for f, q in required_fields if not interpreted.get(f)]
        if missing:
            questions = '\n'.join([f'- {q}' for q in missing])
            # Build a sanitized summary of the interpreted fields (no long text)
            safe_items = []
            try:
                import json as _json
                for k, v in (interpreted or {}).items():
                    if isinstance(v, str):
                        snippet = v.strip().replace('\n', ' ')[:120]
                        safe_items.append(f"{k}: \"{snippet}\"")
                    else:
                        safe_items.append(f"{k}: {type(v).__name__}")
                safe_summary = _json.dumps(safe_items, ensure_ascii=False)
            except Exception:
                safe_summary = str(list((interpreted or {}).keys()))

            # Ask the LLM to produce conversational, field-specific questions with short examples
            return f"""You are a skilled product consultant conducting a discovery interview. Your job is to ask focused, conversational questions to clarify missing or incomplete information about the business idea.

                Current Idea Profile:
                {interpreted}

                MISSING REQUIRED INFORMATION:
                {questions}

                YOUR TASK:
                Generate a JSON array of follow-up questions to fill the gaps. Each question should:
                1. Be conversational and natural (not robotic)
                2. Ask for ONE specific piece of information only
                3. Include a brief example in parentheses to guide the user
                4. Keep it under 20 words
                5. Use second-person perspective ("What", "How", "Why" - address the user directly)

                IMPORTANT RULES:
                - Output ONLY a JSON array of strings - nothing else
                - If information is already complete, return an empty array: []
                - Do not include explanations, code fences, or any text outside the JSON array
                - Order questions by importance/relevance

                EXAMPLE OUTPUT:
                ["What specifically is the main problem your users face? (e.g., 'they spend 2 hours daily on paperwork')", "Who is your primary target user? (e.g., 'small business owners aged 30-55')", "How will users access your product? (e.g., 'web app', 'mobile app', 'desktop software')"]

                Now generate the clarifying questions as a JSON array:"""
                # Fallback to generic prompt if nothing is missing

        return f"""You are a product consultant reviewing a well-defined business idea. Assess whether any key details could be enhanced or clarified for a better understanding.

            Current Idea:
            {interpreted}

            Generate a JSON array of optional clarifying questions to deepen understanding. Focus on:
            - Market dynamics and competitive positioning
            - Specific use cases and user workflows
            - Revenue and growth strategy
            - Technical or operational considerations

            Output ONLY a JSON array of strings. If the idea is sufficiently detailed, return [].
            """

    # ------------------------------------------------------------------
    # 3) Reviewer (market review)
    # ------------------------------------------------------------------

    def review_prompt(
        self,
        interpreted: Dict[str, Any],
        rag_results: List[Dict[str, Any]],
    ) -> str:
        """
        Parameters
        ----------
        interpreted : dict
            Idea JSON.
        rag_results : list of dict
            Retrieved similar tools / companies from vector DB.

        Returns
        -------
        str
            Prompt for generating a short market analysis.
        """
        return f"""You are a seasoned market analyst and product strategist. Analyze this business idea against existing competitors and market dynamics.

            THE IDEA:
            {interpreted}

            SIMILAR EXISTING SOLUTIONS (from market research):
            {rag_results}

            YOUR ANALYSIS (provide exactly this structure):

            1. MARKET POSITIONING (2-3 sentences):
            - Where does this fit in the existing market landscape?
            - Is this incremental innovation or a new category?

            2. COMPETITIVE OVERLAPS (2-3 sentences):
            - What existing solutions address similar needs?
            - What features/benefits overlap?

            3. MARKET GAPS & OPPORTUNITIES (2-3 sentences):
            - What needs are NOT currently met?
            - Where could this idea have a unique advantage?

            4. RISK ASSESSMENT (2-3 sentences):
            - What are the main barriers to success?
            - Any red flags or market headwinds?

            TONE & STYLE:
            - Be analytical and specific (cite actual competitors)
            - Use clear, professional language
            - Avoid hype - be realistic and balanced
            - Focus on facts and market dynamics, not opinions

            Write your analysis in plain English with clear section headers."""

    # ------------------------------------------------------------------
    # 4) MiniReview (differentiation)
    # ------------------------------------------------------------------

    def mini_review_prompt(
    self,
    interpreted: Dict[str, Any],
    rag_results: List[Dict[str, Any]],
    analysis: Optional[str],
    ) -> str:
        """
        Parameters
        ----------
        interpreted : dict
            Idea JSON.
        rag_results : list of dict
            Retrieved similar tools / companies.
        analysis : str or None
            Market analysis text from ReviewerAgent.

        Returns
        -------
        str
            Prompt for a short competitor comparison mini-review.
        """
        return f"""You are a product strategist conducting a competitive differentiation analysis. Your goal is to identify how this idea stands out in the market.

            THE IDEA:
            {interpreted}

            COMPETITIVE LANDSCAPE:
            {rag_results}

            MARKET CONTEXT:
            {analysis}

            YOUR TASK - PROVIDE EXACTLY:

            1. COMPETITIVE POSITIONING (2-3 sentences):
            - How is this similar to existing solutions?
            - What category does it compete in?

            2. KEY DIFFERENTIATORS (3-5 specific points):
            - Unique features or approaches
            - Superior execution or focus
            - Better targeting or pricing
            - Innovation in user experience
            Example format: "- Lower price point (30% cheaper than competitors)"

            3. COMPETITIVE ADVANTAGES (2-3 points):
            - Core strengths that competitors lack
            - Defensible market position
            Example: "- Exclusive partnerships with [industry]"

            4. RISKS & CHALLENGES (2-3 points):
            - What could prevent market adoption?
            - Barriers competitors could exploit
            - Any red flags about the market?
            Example: "- High customer acquisition cost in this category"

            QUALITY REQUIREMENTS:
            - Be specific and actionable (not vague claims)
            - Base analysis on actual competitive context provided
            - Acknowledge strengths AND weaknesses
            - Use realistic, conservative estimates

            Format your response as structured text with clear headers and bullet points."""


    # ------------------------------------------------------------------
    # 5) Strategist (SWOT / strategy)
    # ------------------------------------------------------------------

    def strategy_prompt(self, interpreted: Dict[str, Any]) -> str:
        """
        Parameters
        ----------
        interpreted : dict
            Idea JSON for a brand-new idea (no strong matches in DB).

        Returns
        -------
        str
            Prompt for generating a SWOT or strategic view with research and action plan.
        """
        return f"""You are a seasoned startup strategist and business consultant. Your task is to develop a comprehensive go-to-market strategy for this business idea.

            THE IDEA:
            {interpreted}

            STRATEGIC FRAMEWORK - PROVIDE DETAILED ANALYSIS FOR EACH SECTION:

            ## 1. MARKET OPPORTUNITY & SIZING
            - Total Addressable Market (TAM) - estimate in $ or # of users
            - Serviceable Addressable Market (SAM) - realistic initial target
            - Key customer segments and their pain points (be specific)
            - Current market trends that favor or hinder this idea
            - Barriers to entry (technical, regulatory, capital requirements)

            ## 2. COMPETITIVE & ADJACENT LANDSCAPE
            - Direct competitors and their positioning
            - Indirect competitors (adjacent products solving similar problems)
            - What makes this idea unique vs. the competition
            - Potential new entrants in 2-3 years
            - Why now is the right time to build this

            ## 3. SWOT ANALYSIS (Be Specific)
            
            STRENGTHS (Your unique advantages):
            - What your team/idea does better than competitors
            - Unique capabilities or resources
            - Market timing advantages
            
            WEAKNESSES (Honest assessment):
            - Gaps in product/team/resources
            - Limitations compared to competitors
            - Execution risks
            
            OPPORTUNITIES (Market growth vectors):
            - Adjacent markets to expand into
            - Product extensions or new features
            - Strategic partnerships possible
            - Emerging technologies to leverage
            
            THREATS (External risks):
            - Competitive threats
            - Regulatory/compliance risks
            - Market saturation concerns
            - Technology disruption risks

            ## 4. GO-TO-MARKET & EXECUTION STRATEGY
            
            PHASE 1 - VALIDATION (Weeks 1-8):
            - [ ] Customer discovery interviews (target # of interviews)
            - [ ] Market size validation research
            - [ ] Competitor analysis deep-dive
            - [ ] Technical feasibility assessment
            - [ ] Success metric: [specific validation goal]
            
            PHASE 2 - MVP DEVELOPMENT (Months 2-3):
            - [ ] Core features to build first
            - [ ] Technology stack recommendations
            - [ ] Resources needed (team, budget)
            - [ ] Timeline and milestones
            - [ ] Success metric: MVP launch with X beta users
            
            PHASE 3 - MARKET ENTRY (Months 4-6):
            - [ ] Go-to-market channels (direct sales, marketing, partnerships)
            - [ ] Customer acquisition strategy
            - [ ] Pricing strategy and model
            - [ ] Initial growth targets
            - [ ] Success metric: acquire X customers, $X MRR
            
            PHASE 4 - SCALING (Months 7-12):
            - [ ] Product-market fit validation (NPS, retention targets)
            - [ ] Team expansion priorities
            - [ ] Funding requirements and use of capital
            - [ ] Geographic or segment expansion opportunities

            ## 5. KEY SUCCESS METRICS & MILESTONES
            - List 3-5 critical KPIs to track
            - Define success criteria for each phase
            - Identify decision points where you'd pivot or double-down
            - Timeline: 30, 60, 90 days, 6 months, 12 months

            ## 6. FUNDING & RESOURCE REQUIREMENTS
            - Estimated capital needed to reach milestones
            - Key hires required by phase
            - Budget allocation: R&D, marketing, operations
            - Assumptions about burn rate and runway

            ## TONE & STYLE:
            - Be thorough but concise (concrete details, not fluff)
            - Use realistic estimates based on industry benchmarks
            - Identify key assumptions and risks
            - Provide actionable recommendations, not generic advice
            - Support claims with logic and reasoning

            Format your response with clear section headers and bullet points for easy scanning."""

    # ------------------------------------------------------------------
    # 6) Summarizer (final answer)
    # ------------------------------------------------------------------

    def summarizer_prompt(
        self,
        interpreted: Dict[str, Any],
        analysis: Optional[str],
        mini_review: Any,
        strategy: Any,
    ) -> str:
        """
        Parameters
        ----------
        interpreted : dict
            The structured idea.
        analysis : str or None
            Market review text (existing solutions).
        mini_review : any
            Mini-review information (text or dict).
        strategy : any
            Strategic / SWOT info (text or dict), may be None.

        Returns
        -------
        str
            Prompt for producing the final answer for the end user.
        """
        return f"""You are a business consultant synthesizing a comprehensive analysis of a business idea. Your goal is to create a clear, actionable executive summary for the founder.

            BUSINESS IDEA DETAILS:
            {interpreted}

            MARKET ANALYSIS:
            {analysis}

            STRATEGIC ROADMAP:
            {strategy}

            YOUR TASK - STRUCTURE YOUR RESPONSE EXACTLY AS FOLLOWS (use plain text with line breaks, NO markdown formatting):

            EXECUTIVE SUMMARY
            Provide a compelling 3-4 sentence overview:
            - What the idea is
            - Why it matters
            - Why NOW is the right time
            - Key competitive advantage

            OPPORTUNITY ASSESSMENT
            - Market size and growth potential
            - Target customer profile (be specific)
            - Primary value proposition (1-2 sentences)

            COMPETITIVE POSITION
            - How you differentiate from alternatives
            - Key advantages vs. competitors (2-3 points)
            - Realistic assessment of challenges

            GO-TO-MARKET STRATEGY
            1. Initial Target Market: [specific segment]
            2. Primary Validation Focus: [most critical assumption to test]
            3. Quick Wins (0-4 weeks): [2-3 specific actions]
            4. MVP Focus (1-3 months): [core features only]

            CRITICAL SUCCESS FACTORS
            List 3-5 things that MUST go right for this to succeed:
            1. [Specific requirement with explanation]
            2. [Specific requirement with explanation]
            (etc.)

            IMMEDIATE NEXT STEPS (Ranked by Priority)
            For the founder RIGHT NOW, in this order:
            1. [Action] - Expected outcome: [specific result]
            2. [Action] - Expected outcome: [specific result]
            3. [Action] - Expected outcome: [specific result]
            4. [Action] - Expected outcome: [specific result]

            RISK ASSESSMENT
            Identify 2-3 key risks with suggested mitigation:
            - Risk: [specific threat]
            Mitigation: [concrete action to address]

            OVERALL VERDICT
            Conclude with an honest, balanced assessment:
            - Viability (High/Medium/Low) with 1-2 sentence justification
            - Key assumptions to validate before major investment
            - Recommended pivot points if validation fails

            IMPORTANT GUIDELINES:
            - Be specific: cite actual data, markets, competitors from previous analysis
            - Be honest: acknowledge both strengths and significant challenges
            - Be actionable: founder should know exactly what to do next
            - Be concise: prioritize most important insights
            - Avoid hype: use conservative, realistic language
            - Focus on what matters: execution and market-fit, not perfect planning
            - DO NOT use markdown formatting (no #, ##, *bold*, etc.)
            - Use plain text with clear line breaks between sections
            - Use dashes (-) and numbers (1., 2., etc.) for lists, but no markdown bold
            - Don't repeat information: each section should add new insights
            - Don't give what I told you, create something new and valuable which includes don't repeat the prompt

            Create a compelling, professional summary that motivates action while grounding the founder in reality."""
