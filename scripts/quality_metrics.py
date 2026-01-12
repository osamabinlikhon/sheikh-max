#!/usr/bin/env python3
"""
Security-First Quality Metrics for Sheikh-Max
Evaluates the quality of interleaved thinking with emphasis on security considerations.
"""

import re
from typing import Dict, List, Any, Callable


# Security Priority Metrics for evaluating <think> content
SECURITY_PRIORITY_METRICS: Dict[str, Dict[str, Any]] = {
    "input_validation_coverage": {
        "weight": 0.25,
        "keywords": [
            'sanitize', 'validate', 'escape', 'xss', 'sql injection',
            'input validation', 'parameter validation', 'whitelist',
            'blacklist', 'regex pattern', 'type check'
        ],
        "critical": True,
        "description": "Comprehensive input validation planning"
    },
    "authentication_authorization": {
        "weight": 0.20,
        "keywords": [
            'auth', 'authentication', 'authorization', 'rbac', 'permissions',
            'jwt', 'oauth', 'session management', 'token', 'credential',
            'login', 'access control', 'role-based'
        ],
        "critical": True,
        "description": "Proper auth mechanisms considered"
    },
    "sensitive_data_handling": {
        "weight": 0.20,
        "keywords": [
            'encryption', 'hashing', 'secrets', 'environment variables',
            'never hardcode', 'secure storage', 'credentials', 'password',
            'api key', 'private key', 'pii', 'sensitive data'
        ],
        "critical": True,
        "description": "Secure handling of sensitive data"
    },
    "error_handling_security": {
        "weight": 0.15,
        "keywords": [
            'error handling', 'exception handling', 'no sensitive info in errors',
            'secure logging', 'graceful degradation', 'try except', 'catch',
            'error message', 'stack trace'
        ],
        "critical": False,
        "description": "Security-conscious error handling"
    },
    "dependency_security": {
        "weight": 0.10,
        "keywords": [
            'dependency check', 'vulnerability scan', 'package security',
            'update dependencies', 'known vulnerabilities', 'cve',
            'security audit', 'outdated'
        ],
        "critical": False,
        "description": "Third-party dependency security assessment"
    },
    "principle_of_least_privilege": {
        "weight": 0.10,
        "keywords": [
            'least privilege', 'minimal permissions', 'role-based access',
            'need-to-know', 'privilege escalation', 'scope', 'restrict'
        ],
        "critical": True,
        "description": "Application of least privilege principle"
    }
}

# Code Quality Metrics
CODE_QUALITY_METRICS: Dict[str, Dict[str, Any]] = {
    "problem_decomposition": {
        "weight": 0.20,
        "keywords": [
            'step', 'first', 'then', 'next', 'finally', 'break down',
            'decompose', 'approach', 'strategy', 'plan'
        ],
        "description": "Clear problem decomposition"
    },
    "edge_case_handling": {
        "weight": 0.20,
        "keywords": [
            'edge case', 'corner case', 'boundary', 'empty', 'null', 'none',
            'zero', 'negative', 'overflow', 'underflow', 'special case'
        ],
        "description": "Edge case consideration"
    },
    "algorithm_analysis": {
        "weight": 0.15,
        "keywords": [
            'time complexity', 'space complexity', 'o(n)', 'o(log n)',
            'big o', 'efficient', 'optimize', 'performance'
        ],
        "description": "Algorithm complexity analysis"
    },
    "code_structure": {
        "weight": 0.15,
        "keywords": [
            'function', 'class', 'module', 'interface', 'abstract',
            'encapsulation', 'separation of concerns', 'single responsibility'
        ],
        "description": "Code structure planning"
    },
    "testing_consideration": {
        "weight": 0.15,
        "keywords": [
            'test', 'unit test', 'integration test', 'mock', 'assert',
            'verify', 'validate', 'coverage'
        ],
        "description": "Testing strategy"
    },
    "documentation": {
        "weight": 0.15,
        "keywords": [
            'docstring', 'comment', 'document', 'explain', 'describe',
            'readme', 'type hint', 'annotation'
        ],
        "description": "Documentation planning"
    }
}


def check_keywords(content: str, keywords: List[str]) -> bool:
    """Check if any keywords are present in content."""
    content_lower = content.lower()
    return any(kw.lower() in content_lower for kw in keywords)


def calculate_metric_score(content: str, keywords: List[str]) -> float:
    """Calculate score based on keyword presence and density."""
    content_lower = content.lower()
    matches = sum(1 for kw in keywords if kw.lower() in content_lower)
    # Score based on percentage of keywords found (max 1.0)
    return min(1.0, matches / max(len(keywords) * 0.3, 1))


def calculate_security_score(thinking_content: str) -> float:
    """
    Calculate overall security score for thinking content.
    
    Args:
        thinking_content: The content inside <think> tags
        
    Returns:
        float: Security score between 0.0 and 1.0
    """
    total_score = 0.0
    
    for metric_name, metric_config in SECURITY_PRIORITY_METRICS.items():
        keywords = metric_config["keywords"]
        weight = metric_config["weight"]
        score = calculate_metric_score(thinking_content, keywords)
        total_score += score * weight
    
    return total_score


def calculate_code_quality_score(thinking_content: str) -> float:
    """
    Calculate code quality score for thinking content.
    
    Args:
        thinking_content: The content inside <think> tags
        
    Returns:
        float: Quality score between 0.0 and 1.0
    """
    total_score = 0.0
    
    for metric_name, metric_config in CODE_QUALITY_METRICS.items():
        keywords = metric_config["keywords"]
        weight = metric_config["weight"]
        score = calculate_metric_score(thinking_content, keywords)
        total_score += score * weight
    
    return total_score


def has_critical_security_failures(thinking_content: str) -> bool:
    """
    Check for missing critical security considerations.
    
    Args:
        thinking_content: The content inside <think> tags
        
    Returns:
        bool: True if critical security considerations are missing
    """
    critical_failures = []
    
    for metric_name, metric_config in SECURITY_PRIORITY_METRICS.items():
        if metric_config.get("critical", False):
            if not check_keywords(thinking_content, metric_config["keywords"]):
                critical_failures.append(metric_name)
    
    # Return True if more than half of critical metrics are missing
    critical_count = sum(1 for m in SECURITY_PRIORITY_METRICS.values() if m.get("critical"))
    return len(critical_failures) > critical_count / 2


def get_detailed_security_report(thinking_content: str) -> Dict[str, Any]:
    """
    Generate detailed security analysis report.
    
    Args:
        thinking_content: The content inside <think> tags
        
    Returns:
        dict: Detailed report with scores and recommendations
    """
    report = {
        "overall_score": 0.0,
        "metrics": {},
        "critical_issues": [],
        "recommendations": []
    }
    
    total_score = 0.0
    
    for metric_name, metric_config in SECURITY_PRIORITY_METRICS.items():
        keywords = metric_config["keywords"]
        weight = metric_config["weight"]
        score = calculate_metric_score(thinking_content, keywords)
        weighted_score = score * weight
        total_score += weighted_score
        
        report["metrics"][metric_name] = {
            "score": score,
            "weighted_score": weighted_score,
            "weight": weight,
            "critical": metric_config.get("critical", False),
            "description": metric_config["description"]
        }
        
        # Add to critical issues if critical metric has low score
        if metric_config.get("critical", False) and score < 0.3:
            report["critical_issues"].append({
                "metric": metric_name,
                "description": metric_config["description"],
                "recommendation": f"Consider adding {metric_name.replace('_', ' ')} to your thinking process"
            })
    
    report["overall_score"] = total_score
    
    # Generate recommendations
    if total_score < 0.5:
        report["recommendations"].append(
            "Security thinking is below threshold. Consider threat modeling before implementation."
        )
    if not check_keywords(thinking_content, ["validate", "sanitize", "escape"]):
        report["recommendations"].append(
            "Add input validation strategy to your thinking process."
        )
    if not check_keywords(thinking_content, ["encrypt", "hash", "secret"]):
        report["recommendations"].append(
            "Consider data protection mechanisms for sensitive information."
        )
    
    return report


def security_weighted_loss(thinking_content: str, base_loss: float) -> float:
    """
    Apply security-aware weighting to loss function.
    Critical security failures result in higher loss penalties.
    
    Args:
        thinking_content: The content inside <think> tags
        base_loss: The original loss value
        
    Returns:
        float: Adjusted loss with security penalty
    """
    security_score = calculate_security_score(thinking_content)
    penalty_factor = 1.0
    
    # Critical security failures get 3x penalty
    if security_score < 0.3 and has_critical_security_failures(thinking_content):
        penalty_factor = 3.0
    # Moderate security issues get 1.5x penalty
    elif security_score < 0.7:
        penalty_factor = 1.5
    
    return base_loss * penalty_factor


def extract_thinking_content(response: str) -> str:
    """
    Extract content from <think> tags.
    
    Args:
        response: Full model response
        
    Returns:
        str: Content inside <think> tags, or empty string if not found
    """
    match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
    return match.group(1).strip() if match else ""


def evaluate_response(response: str) -> Dict[str, Any]:
    """
    Comprehensive evaluation of a model response.
    
    Args:
        response: Full model response with <think> tags
        
    Returns:
        dict: Evaluation results including security and quality scores
    """
    thinking_content = extract_thinking_content(response)
    
    if not thinking_content:
        return {
            "has_thinking": False,
            "security_score": 0.0,
            "quality_score": 0.0,
            "overall_score": 0.0,
            "critical_failures": True,
            "message": "No <think> tags found in response"
        }
    
    security_score = calculate_security_score(thinking_content)
    quality_score = calculate_code_quality_score(thinking_content)
    critical_failures = has_critical_security_failures(thinking_content)
    
    # Overall score: 60% security, 40% quality
    overall_score = security_score * 0.6 + quality_score * 0.4
    
    return {
        "has_thinking": True,
        "security_score": security_score,
        "quality_score": quality_score,
        "overall_score": overall_score,
        "critical_failures": critical_failures,
        "security_report": get_detailed_security_report(thinking_content),
        "thinking_length": len(thinking_content),
        "message": "Evaluation complete"
    }


if __name__ == "__main__":
    # Test with sample thinking content
    sample_thinking = """
    1. **Analyze the Request:** The user wants a login function.
    2. **Security Considerations:**
       - Input validation: Sanitize username and password inputs
       - Authentication: Use secure password hashing (bcrypt)
       - Session management: Generate secure session tokens
       - Error handling: Don't expose sensitive info in error messages
    3. **Edge Cases:**
       - Empty username/password
       - SQL injection attempts
       - Brute force protection
    4. **Implementation:**
       - Validate inputs first
       - Hash password with bcrypt
       - Compare with stored hash
       - Generate JWT token on success
    """
    
    print("🔒 Security Quality Metrics Test")
    print("=" * 50)
    
    security_score = calculate_security_score(sample_thinking)
    quality_score = calculate_code_quality_score(sample_thinking)
    
    print(f"Security Score: {security_score:.2f}")
    print(f"Quality Score: {quality_score:.2f}")
    print(f"Critical Failures: {has_critical_security_failures(sample_thinking)}")
    
    print("\n📊 Detailed Security Report:")
    report = get_detailed_security_report(sample_thinking)
    for metric, data in report["metrics"].items():
        status = "✅" if data["score"] >= 0.5 else "⚠️" if data["score"] >= 0.3 else "❌"
        print(f"  {status} {metric}: {data['score']:.2f} (weight: {data['weight']})")
    
    if report["recommendations"]:
        print("\n💡 Recommendations:")
        for rec in report["recommendations"]:
            print(f"  • {rec}")
