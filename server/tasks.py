"""
SprintBoard — Task definitions for 15 sprint planning scenarios.

Each task maps to a real-world sprint planning fault type and contains
scenario parameters, a realistic alert message, and safety configuration.
Tasks are ordered by difficulty: easy (1–5) → medium (6–10) → hard (11–15).

Real-world utility: Every task models a planning failure that Scrum Masters
and Engineering Managers encounter regularly. The 15 tasks cover 5 fault
categories — estimation, assignment, dependency, capacity, and process —
providing broad coverage of the agile planning domain.

Difficulty calibration: Easy tasks test single-fault diagnosis (solvable in
3–5 steps). Medium tasks introduce ambiguity or multi-step planning. Hard
tasks present compound faults requiring multi-root-cause analysis. Current
frontier models achieve 0.4–0.7 on compound tasks, leaving significant
headroom for improvement through RL training.

Alert design: Alert messages model production sprint planning scenarios with
observable symptoms only — no root-cause hints. This forces agents to
investigate before acting, mirroring real planning meetings.
"""

from typing import Dict, Any

# ── Alert messages ─────────────────────────────────────────────────
# Alerts give symptoms, never diagnoses. The agent must figure out
# the root cause through investigation, just like a real SM reading
# a sprint health dashboard.

ALERTS = {
    "unestimated_stories": (
        "SCENARIO [Sprint 14 Planning]: The team is starting sprint planning but "
        "3 stories in the backlog have no point estimates. The team cannot commit "
        "to a sprint plan without knowing the effort involved. Please estimate "
        "these stories and plan the sprint."
    ),
    "developer_overload": (
        "SCENARIO [Sprint 14 Mid-Check]: A capacity review shows one team member "
        "is assigned 24 story points while their capacity is only 8 points. Other "
        "team members have unused capacity. The sprint plan needs rebalancing. "
        "Please investigate and fix the assignments."
    ),
    "missing_dependency": (
        "SCENARIO [Sprint 14 Planning]: A story was added to the sprint but its "
        "blocking dependency is not included. The team will be blocked mid-sprint "
        "if this isn't caught now. Please identify the missing dependency and "
        "fix the sprint plan."
    ),
    "scope_creep": (
        "SCENARIO [Sprint 14 Review]: Several stories in the current sprint have "
        "vague, single-line acceptance criteria. Past sprints with unclear scope "
        "consistently resulted in overruns. Please flag the at-risk stories and "
        "add risk notes."
    ),
    "wrong_priority": (
        "SCENARIO [Sprint 14 Planning]: The product owner flagged that critical "
        "P0 stories are not included in the sprint, while lower-priority P2 work "
        "has been added. The sprint plan doesn't align with business priorities. "
        "Please investigate and fix."
    ),
    "velocity_overload": (
        "SCENARIO [Sprint 15 Planning]: The backlog has 18 stories totaling 94 "
        "points. The team's average velocity over the last 5 sprints is 34 points. "
        "The sprint is significantly overloaded. Please investigate velocity "
        "history and trim the sprint to match team capacity."
    ),
    "skill_mismatch": (
        "SCENARIO [Sprint 14 Assignment Review]: A review of assignments shows "
        "that a frontend-only developer has been assigned backend API stories, "
        "and a backend engineer has UI work. These mismatches will cause delays. "
        "Please investigate team skills and reassign appropriately."
    ),
    "epic_decomposition": (
        "SCENARIO [Backlog Refinement]: A large epic worth ~40 story points is "
        "sitting in the backlog. It's too large for a single sprint and needs to "
        "be broken down into 4–6 sprint-sized stories before it can be planned. "
        "Please decompose the epic into actionable user stories."
    ),
    "priority_conflict": (
        "SCENARIO [Sprint 15 Planning]: Two P0 stories have been identified, but "
        "the team only has capacity for one. Both stakeholders insist their story "
        "is more critical. Please investigate both stories, assess impact, and "
        "recommend which to include with justification."
    ),
    "tech_debt_balance": (
        "SCENARIO [Sprint 15 Planning]: The sprint has been loaded entirely with "
        "feature work. The team's bug backlog has grown to 12 items, 3 of which "
        "are P1 severity. Engineering guidelines require at least 20% of sprint "
        "capacity for tech debt. Please rebalance the sprint."
    ),
    "dependency_chain_overload": (
        "SCENARIO [Sprint 15 Planning — COMPOUND]: Two problems exist simultaneously: "
        "(1) A circular dependency chain was detected — Story A depends on B, and B "
        "depends on A. (2) Developer Alice is overloaded at 21 points (capacity 10). "
        "Both issues must be resolved before the sprint can be committed."
    ),
    "pto_velocity_drop": (
        "SCENARIO [Sprint 16 Planning — COMPOUND]: Two problems are compounding: "
        "(1) Senior developer Alice is on PTO for the entire sprint. (2) Team "
        "velocity has been declining over the last 3 sprints (38 → 33 → 28). "
        "The sprint plan doesn't account for either. Please adjust."
    ),
    "cross_team_dependency": (
        "SCENARIO [Sprint 16 Planning — COMPOUND]: The sprint contains stories "
        "that depend on work being done by Team Beta. Team Beta's sprint plan "
        "shows their dependency won't be started until week 2. Additionally, "
        "a skill gap exists — no one on your team can handle the DevOps story. "
        "Please investigate and resolve both coordination issues."
    ),
    "sprint_rescue": (
        "SCENARIO [Sprint 14 Rescue — COMPOUND]: The existing sprint plan has "
        "5 problems simultaneously: an unestimated story, a capacity overload, "
        "a missing dependency, a skill mismatch, and misaligned priorities. "
        "Audit the sprint, identify all issues, and fix them without breaking "
        "other constraints."
    ),
    "full_sprint_planning": (
        "SCENARIO [Sprint 17 Planning — COMPOUND]: You must plan an entire sprint "
        "from scratch. The backlog has 20 stories. You have 4 developers with "
        "different skills and capacities. Stories have dependencies, varying "
        "priorities, and some need estimation. Build a complete, valid sprint plan."
    ),
}


# ── Team member data ───────────────────────────────────────────────
# Realistic team profiles with skills, capacities, and velocity history.

TEAM_MEMBERS = {
    "Alice": {
        "role": "Senior Backend Engineer",
        "capacity": 10,
        "skills": ["python", "java", "postgresql", "api", "microservices"],
    },
    "Bob": {
        "role": "Mid Frontend Engineer",
        "capacity": 8,
        "skills": ["react", "typescript", "css", "html", "nextjs"],
    },
    "Charlie": {
        "role": "Junior Full-Stack Developer",
        "capacity": 5,
        "skills": ["python", "react", "html", "css"],
    },
    "Diana": {
        "role": "Senior Full-Stack Engineer",
        "capacity": 10,
        "skills": ["python", "react", "typescript", "postgresql", "api", "devops", "docker"],
    },
    "Eve": {
        "role": "Mid Backend Engineer",
        "capacity": 8,
        "skills": ["java", "python", "api", "postgresql", "kafka"],
    },
}

VELOCITY_HISTORY = [38, 34, 36, 31, 33]  # Last 5 sprints


# ── Story pool ─────────────────────────────────────────────────────
# Shared story definitions reused across tasks.

STORY_POOL = {
    "101": {
        "title": "User Authentication Redesign",
        "description": "Redesign the authentication flow to support OAuth2 and SSO. "
                       "Includes login, registration, password reset, and session management.",
        "points": 8, "priority": "P0", "type": "feature",
        "skills_required": ["python", "api"],
        "acceptance_criteria": [
            "OAuth2 login flow works end-to-end",
            "SSO integration with company IdP tested",
            "Password reset email sends within 30s",
            "Session timeout after 30min inactivity",
        ],
    },
    "102": {
        "title": "Payment Gateway Integration",
        "description": "Integrate Stripe payment processing for subscription billing. "
                       "Handle card payments, invoicing, and webhook reconciliation.",
        "points": 8, "priority": "P0", "type": "feature",
        "skills_required": ["python", "api"],
        "acceptance_criteria": [
            "Credit card payments process successfully",
            "Webhook events reconcile within 5 minutes",
            "Invoice PDF generation works",
            "Failed payment retry logic implemented",
        ],
    },
    "103": {
        "title": "Dashboard Analytics Widget",
        "description": "Build an analytics dashboard showing key metrics: DAU, revenue, churn.",
        "points": 5, "priority": "P1", "type": "feature",
        "skills_required": ["react", "typescript"],
        "acceptance_criteria": [
            "DAU chart renders with real data",
            "Revenue graph updates in real-time",
            "Churn metric displays with trend arrow",
        ],
    },
    "104": {
        "title": "API Rate Limiter",
        "description": "Implement rate limiting middleware for the public API. "
                       "Support per-user and per-IP limits with configurable windows.",
        "points": 5, "priority": "P1", "type": "feature",
        "skills_required": ["python", "api"],
        "acceptance_criteria": [
            "Rate limits configurable per endpoint",
            "429 responses include retry-after header",
            "Rate limit state persists across restarts",
        ],
    },
    "105": {
        "title": "Database Migration Script",
        "description": "Write migration script to add new columns to the users table "
                       "and backfill existing records. Must be zero-downtime.",
        "points": 3, "priority": "P1", "type": "tech-debt",
        "skills_required": ["postgresql", "python"],
        "dependencies": ["101"],
        "acceptance_criteria": [
            "Migration runs without downtime",
            "All existing records backfilled correctly",
            "Rollback script tested",
        ],
    },
    "106": {
        "title": "Mobile Responsive Header",
        "description": "Fix the site header to be fully responsive on mobile devices.",
        "points": 3, "priority": "P2", "type": "feature",
        "skills_required": ["react", "css"],
        "acceptance_criteria": [
            "Header collapses to hamburger on mobile",
            "Menu slides in from right",
            "No horizontal scroll on any device width",
        ],
    },
    "107": {
        "title": "Error Logging Service",
        "description": "Set up centralized error logging with Sentry integration.",
        "points": 3, "priority": "P1", "type": "tech-debt",
        "skills_required": ["python"],
        "acceptance_criteria": [
            "All unhandled exceptions sent to Sentry",
            "Source maps uploaded for frontend",
            "Alerts configured for error rate spikes",
        ],
    },
    "108": {
        "title": "Search Autocomplete",
        "description": "Add autocomplete suggestions to the main search bar using elasticsearch.",
        "points": 5, "priority": "P2", "type": "feature",
        "skills_required": ["react", "typescript", "api"],
        "acceptance_criteria": [
            "Suggestions appear after 2 characters",
            "Results update within 200ms",
            "Keyboard navigation works",
        ],
    },
    "109": {
        "title": "CI/CD Pipeline Fix",
        "description": "The deployment pipeline has been failing intermittently due to flaky "
                       "integration tests. Fix the test suite and add retry logic.",
        "points": 3, "priority": "P1", "type": "tech-debt",
        "skills_required": ["devops", "python"],
        "acceptance_criteria": [
            "CI pipeline passes 100% of the time",
            "Flaky tests identified and fixed",
            "Retry logic added for network-dependent tests",
        ],
    },
    "110": {
        "title": "User Profile Page",
        "description": "Build the user profile page with avatar upload and settings.",
        "points": 5, "priority": "P2", "type": "feature",
        "skills_required": ["react", "typescript", "api"],
        "dependencies": ["101"],
        "acceptance_criteria": [
            "User can upload and crop avatar",
            "Settings save without page reload",
            "Profile URL is shareable",
        ],
    },
    "111": {
        "title": "Notification System Backend",
        "description": "Build the backend notification system supporting email, push, "
                       "and in-app notifications with user preferences.",
        "points": 8, "priority": "P1", "type": "feature",
        "skills_required": ["python", "api", "kafka"],
        "acceptance_criteria": [
            "Email notifications send within 60s",
            "Push notifications work on mobile",
            "Users can toggle notification types",
            "Notification queue handles 1000 msg/s",
        ],
    },
    "112": {
        "title": "Frontend Unit Test Coverage",
        "description": "Increase frontend test coverage from 45% to 80%.",
        "points": 5, "priority": "P2", "type": "tech-debt",
        "skills_required": ["react", "typescript"],
        "acceptance_criteria": [
            "Coverage reaches 80% on core components",
            "All critical user flows have integration tests",
        ],
    },
    "113": {
        "title": "Admin User Management Panel",
        "description": "Build admin panel for managing users: view, edit roles, deactivate.",
        "points": 5, "priority": "P1", "type": "feature",
        "skills_required": ["react", "python", "api"],
        "dependencies": ["101"],
        "acceptance_criteria": [
            "Admin can search users by email",
            "Role changes take effect immediately",
            "Deactivated users cannot log in",
        ],
    },
    "114": {
        "title": "API Documentation Generator",
        "description": "Set up automatic OpenAPI spec generation from code annotations.",
        "points": 2, "priority": "P2", "type": "tech-debt",
        "skills_required": ["python", "api"],
        "acceptance_criteria": [
            "Swagger UI auto-generates from decorators",
            "All endpoints documented with examples",
        ],
    },
    "115": {
        "title": "File Upload Service",
        "description": "Implement S3-backed file upload with virus scanning and size limits.",
        "points": 5, "priority": "P1", "type": "feature",
        "skills_required": ["python", "api", "devops"],
        "dependencies": ["101"],
        "acceptance_criteria": [
            "Files upload to S3 with presigned URLs",
            "Virus scan runs before storage",
            "Max file size configurable per tenant",
        ],
    },
    "116": {
        "title": "Database Read Replica Setup",
        "description": "Configure read replicas and route read-heavy queries to them.",
        "points": 8, "priority": "P1", "type": "tech-debt",
        "skills_required": ["postgresql", "devops", "python"],
        "acceptance_criteria": [
            "Read replica syncs within 1s",
            "Read queries automatically routed",
            "Failover tested and documented",
        ],
    },
    "117": {
        "title": "Dark Mode Toggle",
        "description": "Add dark mode support across the entire frontend application.",
        "points": 3, "priority": "P2", "type": "feature",
        "skills_required": ["react", "css"],
        "acceptance_criteria": [
            "Toggle persists across sessions",
            "All components respect dark theme",
            "No flash of light theme on load",
        ],
    },
    "118": {
        "title": "Webhook Delivery System",
        "description": "Build outbound webhook system for third-party integrations.",
        "points": 8, "priority": "P1", "type": "feature",
        "skills_required": ["python", "api", "kafka"],
        "dependencies": ["111"],
        "acceptance_criteria": [
            "Webhooks deliver within 5 seconds",
            "Failed deliveries retry with exponential backoff",
            "Webhook logs accessible in admin panel",
            "HMAC signature verification documented",
        ],
    },
    "119": {
        "title": "Performance Monitoring Setup",
        "description": "Integrate APM tools for backend and frontend performance tracking.",
        "points": 3, "priority": "P2", "type": "tech-debt",
        "skills_required": ["devops", "python"],
        "acceptance_criteria": [
            "Backend traces appear in APM dashboard",
            "Frontend Core Web Vitals tracked",
            "Alerting for p99 > 2s configured",
        ],
    },
    "120": {
        "title": "Multi-language Support (i18n)",
        "description": "Add internationalization framework and translate UI to Spanish and French.",
        "points": 8, "priority": "P2", "type": "feature",
        "skills_required": ["react", "typescript"],
        "acceptance_criteria": [
            "Language picker in settings",
            "All UI strings extracted to locale files",
            "Spanish and French translations complete",
            "Date/number formats localized",
        ],
    },
}

# ── Epic definitions ───────────────────────────────────────────────

EPIC_POOL = {
    "EP-01": {
        "title": "Multichannel Payment System",
        "description": (
            "Enable users to pay via Credit Card (Stripe), PayPal, Apple Pay, "
            "and bank transfer with secure checkout, invoice generation, "
            "subscription management, and webhook reconciliation."
        ),
        "total_points": 40,
        "ideal_decomposition": [
            "Stripe credit card payment integration",
            "PayPal checkout flow",
            "Apple Pay / Google Pay support",
            "Invoice generation and PDF export",
            "Subscription billing and recurring charges",
            "Payment webhook reconciliation pipeline",
        ],
        "required_keywords": ["stripe", "paypal", "apple", "invoice", "subscription", "webhook"],
    },
}

# ── Bug backlog (for tech debt tasks) ──────────────────────────────

BUG_BACKLOG = [
    {"id": "BUG-01", "title": "Login fails on Safari", "priority": "P1", "points": 3,
     "skills_required": ["react", "css"]},
    {"id": "BUG-02", "title": "Payment timeout on slow connections", "priority": "P1", "points": 5,
     "skills_required": ["python", "api"]},
    {"id": "BUG-03", "title": "Dashboard crashes with >1000 users", "priority": "P1", "points": 5,
     "skills_required": ["react", "typescript"]},
    {"id": "BUG-04", "title": "Email notifications delayed >5min", "priority": "P2", "points": 3,
     "skills_required": ["python", "kafka"]},
    {"id": "BUG-05", "title": "Search returns stale results", "priority": "P2", "points": 3,
     "skills_required": ["python", "api"]},
    {"id": "BUG-06", "title": "Mobile menu z-index overlap", "priority": "P2", "points": 2,
     "skills_required": ["react", "css"]},
    {"id": "BUG-07", "title": "API 500 on empty request body", "priority": "P2", "points": 2,
     "skills_required": ["python", "api"]},
    {"id": "BUG-08", "title": "Timezone offset in reports", "priority": "P2", "points": 3,
     "skills_required": ["python"]},
    {"id": "BUG-09", "title": "File upload fails >10MB", "priority": "P2", "points": 3,
     "skills_required": ["python", "api"]},
    {"id": "BUG-10", "title": "Rate limiter not resetting", "priority": "P2", "points": 2,
     "skills_required": ["python"]},
    {"id": "BUG-11", "title": "Dark mode text contrast issue", "priority": "P2", "points": 1,
     "skills_required": ["react", "css"]},
    {"id": "BUG-12", "title": "Memory leak in WebSocket handler", "priority": "P1", "points": 5,
     "skills_required": ["python", "api"]},
]


TASK_REGISTRY: Dict[str, Dict[str, Any]] = {
    # ══════════════════════════════════════════════════════════════
    # EASY (tasks 1–5): Single-fault diagnosis, one clear root cause.
    # Solvable in 2–5 steps. Baseline models score 0.7–1.0.
    # ══════════════════════════════════════════════════════════════
    "task_1": {
        "name": "Unestimated Stories",
        "fault_type": "unestimated_stories",
        "difficulty": "easy",
        "description": (
            "3 stories in the sprint backlog have no point estimates. The team "
            "cannot commit without knowing effort. Estimate these stories using "
            "their descriptions and historical data."
        ),
        "alert": ALERTS["unestimated_stories"],
        "params": {
            "unestimated_story_ids": ["103", "106", "107"],
            "sprint_stories": ["101", "103", "105", "106", "107"],
        },
        "fatal_patterns": ["DELETE_STORY", "CLEAR_SPRINT", "REMOVE_DEVELOPER"],
        "allowed_dangerous": [],
    },
    "task_2": {
        "name": "Developer Overload",
        "fault_type": "developer_overload",
        "difficulty": "easy",
        "description": (
            "Developer Alice has been assigned 24 story points but her capacity "
            "is only 10. Other developers have unused capacity. Investigate and "
            "redistribute the workload."
        ),
        "alert": ALERTS["developer_overload"],
        "params": {
            "overloaded_dev": "Alice",
            "overloaded_points": 24,
            "overloaded_capacity": 10,
            "sprint_stories": ["101", "102", "104", "105"],
            "initial_assignments": {
                "101": "Alice",  # 8pts
                "102": "Alice",  # 8pts
                "104": "Alice",  # 5pts
                "105": "Alice",  # 3pts → total 24
            },
        },
        "fatal_patterns": ["DELETE_STORY", "CLEAR_SPRINT", "REMOVE_DEVELOPER"],
        "allowed_dangerous": [],
    },
    "task_3": {
        "name": "Missing Dependency",
        "fault_type": "missing_dependency",
        "difficulty": "easy",
        "description": (
            "Story US-105 (Database Migration) is in the sprint, but its "
            "dependency US-101 (Auth Redesign) is not. The team will be blocked. "
            "Add the missing dependency to the sprint."
        ),
        "alert": ALERTS["missing_dependency"],
        "params": {
            "blocked_story": "105",
            "missing_dep": "101",
            "sprint_stories": ["103", "105", "106", "107"],
        },
        "fatal_patterns": ["DELETE_STORY", "CLEAR_SPRINT", "REMOVE_DEVELOPER"],
        "allowed_dangerous": [],
    },
    "task_4": {
        "name": "Scope Creep Risk",
        "fault_type": "scope_creep",
        "difficulty": "easy",
        "description": (
            "Several stories have vague, single-line acceptance criteria. "
            "Past sprints with unclear scope led to 40% overruns. Flag the "
            "risky stories."
        ),
        "alert": ALERTS["scope_creep"],
        "params": {
            "sprint_stories": ["101", "103", "106", "108", "114"],
            "vague_stories": ["114"],  # Only has 2 ACs, descriptions are thin
            "vague_story_ids_ground_truth": ["114"],
        },
        "fatal_patterns": ["DELETE_STORY", "CLEAR_SPRINT", "REMOVE_DEVELOPER"],
        "allowed_dangerous": [],
    },
    "task_5": {
        "name": "Wrong Priority Alignment",
        "fault_type": "wrong_priority",
        "difficulty": "easy",
        "description": (
            "P0 stories US-101 and US-102 are not in the sprint, but P2 stories "
            "106 and US-117 are. The sprint doesn't align with business priorities. "
            "Fix the priority alignment."
        ),
        "alert": ALERTS["wrong_priority"],
        "params": {
            "sprint_stories": ["106", "108", "112", "117"],
            "missing_p0s": ["101", "102"],
            "low_priority_in_sprint": ["106", "117"],
        },
        "fatal_patterns": ["DELETE_STORY", "CLEAR_SPRINT", "REMOVE_DEVELOPER"],
        "allowed_dangerous": [],
    },

    # ══════════════════════════════════════════════════════════════
    # MEDIUM (tasks 6–10): Multi-step investigation, ambiguity.
    # Typical resolution: 6–12 steps. Baseline models score 0.4–0.9.
    # ══════════════════════════════════════════════════════════════
    "task_6": {
        "name": "Velocity Overload",
        "fault_type": "velocity_overload",
        "difficulty": "medium",
        "description": (
            "Sprint has 18 stories totaling 94 points but team velocity is ~34. "
            "Investigate velocity history and trim sprint to match capacity."
        ),
        "alert": ALERTS["velocity_overload"],
        "params": {
            "sprint_stories": [
                "101", "102", "103", "104", "105",
                "106", "107", "108", "109", "110",
                "111", "112", "113", "114", "115",
                "116", "117", "118",
            ],
            "total_points": 94,
            "velocity_avg": 34,
        },
        "fatal_patterns": ["DELETE_STORY", "CLEAR_SPRINT", "REMOVE_DEVELOPER"],
        "allowed_dangerous": [],
    },
    "task_7": {
        "name": "Skill Mismatch",
        "fault_type": "skill_mismatch",
        "difficulty": "medium",
        "description": (
            "Frontend stories are assigned to backend-only developers and vice "
            "versa. Check team skills and reassign appropriately."
        ),
        "alert": ALERTS["skill_mismatch"],
        "params": {
            "sprint_stories": ["103", "104", "106", "107"],
            "initial_assignments": {
                "103": "Alice",   # Dashboard (react) → Alice (backend-only)
                "104": "Bob",     # API Rate Limiter (python) → Bob (frontend-only)
                "106": "Eve",     # Mobile Header (react/css) → Eve (backend)
                "107": "Bob",     # Error Logging (python) → Bob (frontend-only)
            },
            "correct_skill_assignments": {
                "103": ["Bob", "Charlie", "Diana"],  # needs react
                "104": ["Alice", "Diana", "Eve"],     # needs python/api
                "106": ["Bob", "Charlie", "Diana"],   # needs react/css
                "107": ["Alice", "Charlie", "Diana", "Eve"],  # needs python
            },
        },
        "fatal_patterns": ["DELETE_STORY", "CLEAR_SPRINT", "REMOVE_DEVELOPER"],
        "allowed_dangerous": [],
    },
    "task_8": {
        "name": "Epic Decomposition",
        "fault_type": "epic_decomposition",
        "difficulty": "medium",
        "description": (
            "Epic EP-01 (Multichannel Payment System, ~40pts) needs to be broken "
            "into 4–6 sprint-sized stories before planning."
        ),
        "alert": ALERTS["epic_decomposition"],
        "params": {
            "epic_id": "EP-01",
            "epic": EPIC_POOL["EP-01"],
            "min_subtasks": 4,
            "max_subtasks": 8,
        },
        "fatal_patterns": ["DELETE_STORY", "CLEAR_SPRINT", "REMOVE_DEVELOPER"],
        "allowed_dangerous": [],
    },
    "task_9": {
        "name": "Priority Conflict Resolution",
        "fault_type": "priority_conflict",
        "difficulty": "medium",
        "description": (
            "Two P0 stories but only capacity for one. Investigate both, "
            "assess impact, and recommend which to include with justification."
        ),
        "alert": ALERTS["priority_conflict"],
        "params": {
            "conflicting_stories": ["101", "102"],
            "sprint_stories": ["103", "106", "107"],
            "remaining_capacity": 10,
        },
        "fatal_patterns": ["DELETE_STORY", "CLEAR_SPRINT", "REMOVE_DEVELOPER"],
        "allowed_dangerous": [],
    },
    "task_10": {
        "name": "Tech Debt Balance",
        "fault_type": "tech_debt_balance",
        "difficulty": "medium",
        "description": (
            "Sprint is 100% feature work. Bug backlog has 12 items including "
            "3 P1 bugs. At least 20% of capacity should be tech debt/bugs."
        ),
        "alert": ALERTS["tech_debt_balance"],
        "params": {
            "sprint_stories": ["101", "102", "103", "108", "110", "111"],
            "total_sprint_points": 39,
            "tech_debt_target_pct": 0.20,
            "p1_bugs": ["BUG-01", "BUG-02", "BUG-03"],
        },
        "fatal_patterns": ["DELETE_STORY", "CLEAR_SPRINT", "REMOVE_DEVELOPER"],
        "allowed_dangerous": [],
    },

    # ══════════════════════════════════════════════════════════════
    # HARD (tasks 11–15): Compound faults requiring multi-root-cause
    # analysis. Two+ simultaneous problems interact. Fixing only one
    # yields partial credit. Current frontier models achieve 0.3–0.7.
    # ══════════════════════════════════════════════════════════════
    "task_11": {
        "name": "Compound: Dependency Chain + Overload",
        "fault_type": "dependency_chain_overload",
        "difficulty": "hard",
        "description": (
            "Two problems exist simultaneously: (1) Circular dependency between "
            "105 and US-113. (2) Alice overloaded at 21pts (capacity 10). "
            "Both issues must be resolved."
        ),
        "alert": ALERTS["dependency_chain_overload"],
        "params": {
            "sprint_stories": ["101", "102", "104", "105", "113"],
            "circular_deps": {"105": "113", "113": "105"},
            "overloaded_dev": "Alice",
            "initial_assignments": {
                "101": "Alice",  # 8pts
                "102": "Alice",  # 8pts
                "104": "Alice",  # 5pts → total 21
                "105": "Diana",
                "113": "Diana",
            },
        },
        "fatal_patterns": ["DELETE_STORY", "CLEAR_SPRINT", "REMOVE_DEVELOPER"],
        "allowed_dangerous": [],
    },
    "task_12": {
        "name": "Compound: PTO + Velocity Drop",
        "fault_type": "pto_velocity_drop",
        "difficulty": "hard",
        "description": (
            "Two problems are compounding: (1) Alice is on PTO for the entire "
            "sprint. (2) Velocity has been declining (38→33→28). Sprint plan "
            "doesn't account for either."
        ),
        "alert": ALERTS["pto_velocity_drop"],
        "params": {
            "sprint_stories": ["101", "102", "103", "104", "105",
                                "107", "108"],
            "pto_developer": "Alice",
            "velocity_history_decline": [38, 35, 33, 30, 28],
            "initial_assignments": {
                "101": "Alice",  # Alice is on PTO!
                "102": "Eve",
                "103": "Bob",
                "104": "Alice",  # Alice is on PTO!
                "105": "Diana",
                "107": "Charlie",
                "108": "Bob",
            },
        },
        "fatal_patterns": ["DELETE_STORY", "CLEAR_SPRINT", "REMOVE_DEVELOPER"],
        "allowed_dangerous": [],
    },
    "task_13": {
        "name": "Compound: Cross-Team + Skill Gap",
        "fault_type": "cross_team_dependency",
        "difficulty": "hard",
        "description": (
            "Sprint has stories depending on Team Beta's work (not starting "
            "until week 2). Additionally, US-109 (DevOps) needs devops skills "
            "that only Diana has, but she's fully loaded."
        ),
        "alert": ALERTS["cross_team_dependency"],
        "params": {
            "sprint_stories": ["104", "107", "109", "115"],
            "cross_team_stories": ["115"],
            "cross_team_dep": "Team Beta API (starts week 2)",
            "skill_gap_story": "109",
            "skill_gap_skill": "devops",
            "only_capable_dev": "Diana",
            "initial_assignments": {
                "104": "Alice",
                "107": "Charlie",
                "109": "Eve",    # Eve doesn't have devops skills
                "115": "Alice",  # Depends on Team Beta
            },
        },
        "fatal_patterns": ["DELETE_STORY", "CLEAR_SPRINT", "REMOVE_DEVELOPER"],
        "allowed_dangerous": [],
    },
    "task_14": {
        "name": "Sprint Rescue (5 Problems)",
        "fault_type": "sprint_rescue",
        "difficulty": "hard",
        "description": (
            "Existing sprint plan has 5 simultaneous problems: unestimated story, "
            "capacity overload, missing dependency, skill mismatch, and priority "
            "misalignment. Find and fix all of them."
        ),
        "alert": ALERTS["sprint_rescue"],
        "params": {
            "sprint_stories": ["102", "103", "105", "106", "108",
                                "112", "114"],
            "initial_assignments": {
                "102": "Alice",   # 8pts
                "103": "Alice",   # 5pts - skill mismatch (react→backend dev)
                "105": "Diana",   # 3pts - dependency US-101 missing from sprint
                "106": "Eve",     # 3pts - skill mismatch (css→backend dev)
                "108": "Alice",   # 5pts → Alice total = 18 (overload!)
                "112": "Bob",     # 5pts
                "114": "Charlie", # UNESTIMATED
            },
            "problems": {
                "unestimated": ["114"],
                "overloaded_dev": "Alice",
                "missing_dep": {"105": "101"},
                "skill_mismatch": {"103": "Alice", "106": "Eve"},
                "priority_issue": "P0 US-101 not in sprint but P2 US-112 is",
            },
        },
        "fatal_patterns": ["DELETE_STORY", "CLEAR_SPRINT", "REMOVE_DEVELOPER"],
        "allowed_dangerous": [],
    },
    "task_15": {
        "name": "Full Sprint Planning from Scratch",
        "fault_type": "full_sprint_planning",
        "difficulty": "hard",
        "description": (
            "Plan an entire sprint from scratch. 20 stories in backlog, 4 "
            "developers with different skills and capacities. Stories have "
            "dependencies, varying priorities, and some need estimation. "
            "Build a complete, valid sprint plan."
        ),
        "alert": ALERTS["full_sprint_planning"],
        "params": {
            "backlog_stories": [
                "101", "102", "103", "104", "105",
                "106", "107", "108", "109", "110",
                "111", "112", "113", "114", "115",
                "116", "117", "118", "119", "120",
            ],
            "team": ["Alice", "Bob", "Charlie", "Diana"],
            "target_velocity": 34,
        },
        "fatal_patterns": ["DELETE_STORY", "CLEAR_SPRINT", "REMOVE_DEVELOPER"],
        "allowed_dangerous": [],
    },
}


def get_task(task_id: str) -> Dict[str, Any]:
    """Look up a task by ID. Raises KeyError if not found."""
    if task_id not in TASK_REGISTRY:
        raise KeyError(f"Unknown task_id: {task_id!r}. Available: {list(TASK_REGISTRY.keys())}")
    return TASK_REGISTRY[task_id]


def list_task_ids() -> list:
    """Return all available task IDs."""
    return list(TASK_REGISTRY.keys())
