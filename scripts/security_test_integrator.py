#!/usr/bin/env python3
"""
Security Test Integrator for Sheikh-Max
Generates and runs security-focused tests based on thinking content.
"""

import os
import re
import json
import subprocess
import tempfile
from typing import Dict, List, Any, Optional
from pathlib import Path


class SecurityAwareTestGenerator:
    """Generate security-focused test cases based on thinking content."""
    
    def __init__(self):
        self.security_rules = {
            'input_validation': [
                'SQL injection patterns',
                'XSS attack vectors',
                'Command injection patterns',
                'Path traversal attempts',
                'LDAP injection',
                'XML injection'
            ],
            'auth_tests': [
                'Missing authentication',
                'Weak password requirements',
                'Session fixation vulnerabilities',
                'Insecure token generation',
                'Broken access control'
            ],
            'data_protection': [
                'Hardcoded credentials',
                'Unencrypted sensitive data',
                'Improper error messages exposing data',
                'Insecure logging practices',
                'Sensitive data in URLs'
            ]
        }
        
        self.test_templates = {
            'input_validation': '''
def test_input_validation_{name}():
    """Test input validation for {description}"""
    # SQL injection test
    malicious_inputs = [
        "' OR '1'='1",
        "'; DROP TABLE users; --",
        "1; SELECT * FROM users",
    ]
    for payload in malicious_inputs:
        result = validate_input(payload)
        assert result == False, f"SQL injection not blocked: {{payload}}"
    
    # XSS test
    xss_payloads = [
        "<script>alert('xss')</script>",
        "<img src=x onerror=alert('xss')>",
        "javascript:alert('xss')",
    ]
    for payload in xss_payloads:
        result = sanitize_output(payload)
        assert "<script>" not in result, f"XSS not sanitized: {{payload}}"
''',
            'auth_tests': '''
def test_authentication_{name}():
    """Test authentication for {description}"""
    # Test missing auth
    assert check_auth(None) == False, "Null auth should fail"
    assert check_auth("") == False, "Empty auth should fail"
    
    # Test invalid tokens
    assert check_auth("invalid_token_12345") == False, "Invalid token should fail"
    assert check_auth("expired_token") == False, "Expired token should fail"
    
    # Test authorization
    assert check_authorization("user", "admin_action") == False, "User should not access admin"
''',
            'data_protection': '''
def test_data_protection_{name}():
    """Test data protection for {description}"""
    # Test no sensitive data in logs
    log_output = capture_log_output(lambda: process_sensitive_data("password123"))
    assert "password123" not in log_output, "Password exposed in logs"
    
    # Test no sensitive data in errors
    try:
        raise_error_with_context("secret_api_key_12345")
    except Exception as e:
        assert "secret_api_key" not in str(e), "API key exposed in error"
    
    # Test encryption
    encrypted = encrypt_sensitive("my_secret")
    assert encrypted != "my_secret", "Data not encrypted"
'''
        }
    
    def extract_security_concerns(self, thinking_content: str) -> List[str]:
        """Extract security concerns mentioned in thinking content."""
        concerns = []
        thinking_lower = thinking_content.lower()
        
        security_keywords = {
            'input_validation': ['validate', 'sanitize', 'escape', 'input', 'injection'],
            'authentication': ['auth', 'login', 'session', 'token', 'jwt', 'oauth'],
            'authorization': ['permission', 'role', 'access control', 'rbac'],
            'encryption': ['encrypt', 'hash', 'secret', 'password', 'credential'],
            'error_handling': ['error', 'exception', 'try', 'catch', 'logging']
        }
        
        for concern_type, keywords in security_keywords.items():
            if any(kw in thinking_lower for kw in keywords):
                concerns.append(concern_type)
        
        return concerns
    
    def generate_security_tests_from_thinking(self, thinking_content: str) -> str:
        """Generate security-focused test cases based on thinking content."""
        concerns = self.extract_security_concerns(thinking_content)
        tests = []
        
        tests.append('"""Auto-generated security tests from Sheikh-Max thinking analysis."""\n')
        tests.append('import pytest\n')
        tests.append('from unittest.mock import Mock, patch\n\n')
        
        for i, concern in enumerate(concerns):
            if concern in ['input_validation']:
                tests.append(self.test_templates['input_validation'].format(
                    name=f"case_{i}",
                    description=concern
                ))
            elif concern in ['authentication', 'authorization']:
                tests.append(self.test_templates['auth_tests'].format(
                    name=f"case_{i}",
                    description=concern
                ))
            elif concern in ['encryption']:
                tests.append(self.test_templates['data_protection'].format(
                    name=f"case_{i}",
                    description=concern
                ))
        
        return '\n'.join(tests)
    
    def run_bandit_analysis(self, code_content: str) -> Dict[str, Any]:
        """Run Bandit security scanner on code."""
        results = {
            'score': 1.0,
            'issues': [],
            'high_severity': 0,
            'medium_severity': 0,
            'low_severity': 0
        }
        
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code_content)
                temp_path = f.name
            
            try:
                result = subprocess.run(
                    ['bandit', '-f', 'json', '-q', temp_path],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                if result.stdout:
                    bandit_results = json.loads(result.stdout)
                    issues = bandit_results.get('results', [])
                    
                    for issue in issues:
                        severity = issue.get('issue_severity', 'LOW')
                        results['issues'].append({
                            'severity': severity,
                            'confidence': issue.get('issue_confidence', 'LOW'),
                            'text': issue.get('issue_text', ''),
                            'line': issue.get('line_number', 0)
                        })
                        
                        if severity == 'HIGH':
                            results['high_severity'] += 1
                        elif severity == 'MEDIUM':
                            results['medium_severity'] += 1
                        else:
                            results['low_severity'] += 1
                    
                    # Calculate score (lower issues = higher score)
                    penalty = (
                        results['high_severity'] * 0.3 +
                        results['medium_severity'] * 0.15 +
                        results['low_severity'] * 0.05
                    )
                    results['score'] = max(0.0, 1.0 - penalty)
                    
            except subprocess.TimeoutExpired:
                results['score'] = 0.5
                results['issues'].append({'text': 'Analysis timed out', 'severity': 'UNKNOWN'})
            except FileNotFoundError:
                # Bandit not installed
                results['score'] = 0.5
                results['issues'].append({'text': 'Bandit not installed', 'severity': 'UNKNOWN'})
            except json.JSONDecodeError:
                results['score'] = 0.5
                
        finally:
            if 'temp_path' in locals():
                os.unlink(temp_path)
        
        return results
    
    def check_common_vulnerabilities(self, code_content: str) -> List[Dict[str, str]]:
        """Check for common security vulnerabilities in code."""
        vulnerabilities = []
        
        # Patterns to check
        patterns = {
            'hardcoded_password': (
                r'password\s*=\s*["\'][^"\']+["\']',
                'Potential hardcoded password detected'
            ),
            'hardcoded_secret': (
                r'(api_key|secret|token)\s*=\s*["\'][^"\']+["\']',
                'Potential hardcoded secret detected'
            ),
            'sql_injection': (
                r'execute\s*\(\s*["\'].*%s.*["\']',
                'Potential SQL injection vulnerability (use parameterized queries)'
            ),
            'shell_injection': (
                r'os\.system\s*\(|subprocess\.call\s*\([^,]+shell\s*=\s*True',
                'Potential shell injection vulnerability'
            ),
            'eval_usage': (
                r'\beval\s*\(',
                'Use of eval() is dangerous - consider alternatives'
            ),
            'pickle_usage': (
                r'pickle\.loads?\s*\(',
                'Pickle can execute arbitrary code - use with caution'
            ),
            'weak_crypto': (
                r'(md5|sha1)\s*\(',
                'Weak cryptographic hash detected - use SHA-256 or better'
            ),
            'debug_mode': (
                r'debug\s*=\s*True|DEBUG\s*=\s*True',
                'Debug mode should be disabled in production'
            )
        }
        
        for vuln_name, (pattern, message) in patterns.items():
            matches = re.finditer(pattern, code_content, re.IGNORECASE)
            for match in matches:
                vulnerabilities.append({
                    'type': vuln_name,
                    'message': message,
                    'match': match.group(),
                    'position': match.start()
                })
        
        return vulnerabilities
    
    def calculate_security_test_coverage(self, code_content: str, test_content: str) -> float:
        """Calculate security test coverage."""
        # Extract function names from code
        code_functions = set(re.findall(r'def\s+(\w+)\s*\(', code_content))
        
        # Extract tested functions from test content
        tested_functions = set()
        for match in re.finditer(r'def\s+test_\w+.*?(?=def\s+test_|\Z)', test_content, re.DOTALL):
            test_body = match.group()
            for func in code_functions:
                if func in test_body:
                    tested_functions.add(func)
        
        if not code_functions:
            return 1.0
        
        return len(tested_functions) / len(code_functions)
    
    def run_security_analysis(self, code_content: str) -> Dict[str, Any]:
        """Run comprehensive security analysis on generated code."""
        results = {
            'bandit_score': 0.5,
            'vulnerability_count': 0,
            'vulnerabilities': [],
            'security_coverage': 0.0,
            'critical_issues': [],
            'recommendations': []
        }
        
        # Run Bandit analysis
        bandit_results = self.run_bandit_analysis(code_content)
        results['bandit_score'] = bandit_results['score']
        results['bandit_issues'] = bandit_results['issues']
        
        # Check common vulnerabilities
        vulnerabilities = self.check_common_vulnerabilities(code_content)
        results['vulnerabilities'] = vulnerabilities
        results['vulnerability_count'] = len(vulnerabilities)
        
        # Identify critical issues
        if bandit_results['high_severity'] > 0:
            results['critical_issues'].append(
                f"Found {bandit_results['high_severity']} high-severity security issues"
            )
        
        for vuln in vulnerabilities:
            if vuln['type'] in ['sql_injection', 'shell_injection', 'hardcoded_secret']:
                results['critical_issues'].append(vuln['message'])
        
        # Generate recommendations
        if 'hardcoded' in str(vulnerabilities).lower():
            results['recommendations'].append(
                "Use environment variables for secrets instead of hardcoding"
            )
        if 'injection' in str(vulnerabilities).lower():
            results['recommendations'].append(
                "Use parameterized queries and input validation"
            )
        if bandit_results['score'] < 0.7:
            results['recommendations'].append(
                "Review and fix security issues identified by Bandit"
            )
        
        return results


def main():
    """Test the security integrator."""
    print("🔒 Security Test Integrator")
    print("=" * 50)
    
    # Sample thinking content
    thinking = """
    I need to implement a login function.
    Security considerations:
    - Validate user input to prevent SQL injection
    - Hash passwords using bcrypt
    - Generate secure session tokens
    - Don't expose sensitive info in error messages
    """
    
    # Sample code
    code = '''
def login(username, password):
    # Validate input
    if not username or not password:
        raise ValueError("Invalid credentials")
    
    # Hash password (example - use bcrypt in production)
    import hashlib
    hashed = hashlib.sha256(password.encode()).hexdigest()
    
    # Check credentials (parameterized query)
    query = "SELECT * FROM users WHERE username = ? AND password = ?"
    result = db.execute(query, (username, hashed))
    
    if result:
        return generate_token(username)
    return None
'''
    
    generator = SecurityAwareTestGenerator()
    
    # Generate tests from thinking
    print("\n📝 Generated Security Tests:")
    tests = generator.generate_security_tests_from_thinking(thinking)
    print(tests[:500] + "..." if len(tests) > 500 else tests)
    
    # Run security analysis
    print("\n🔍 Security Analysis Results:")
    results = generator.run_security_analysis(code)
    print(f"  Bandit Score: {results['bandit_score']:.2f}")
    print(f"  Vulnerabilities Found: {results['vulnerability_count']}")
    
    if results['vulnerabilities']:
        print("\n  ⚠️ Vulnerabilities:")
        for vuln in results['vulnerabilities']:
            print(f"    - {vuln['type']}: {vuln['message']}")
    
    if results['critical_issues']:
        print("\n  🚨 Critical Issues:")
        for issue in results['critical_issues']:
            print(f"    - {issue}")
    
    if results['recommendations']:
        print("\n  💡 Recommendations:")
        for rec in results['recommendations']:
            print(f"    - {rec}")


if __name__ == "__main__":
    main()
