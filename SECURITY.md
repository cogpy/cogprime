# Security Policy

## Supported Versions

CogPrime is currently in alpha development. Security updates are provided for the latest version.

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |
| < 0.1   | :x:                |

## Reporting a Vulnerability

If you discover a security vulnerability in CogPrime, please report it responsibly:

1. **Do not** open a public GitHub issue for security vulnerabilities
2. Email the maintainers at d@rzo.io with details of the vulnerability
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Any suggested fixes (optional)

### Response Timeline

- **Initial Response**: Within 48 hours of report
- **Status Update**: Within 7 days with assessment
- **Resolution**: Varies based on severity and complexity

### What to Expect

- Acknowledgment of your report
- Regular updates on progress
- Credit in security advisories (unless you prefer anonymity)
- Notification when the issue is resolved

## Security Considerations

### AI Safety

CogPrime is an AGI research project. When deploying or extending:

- Review cognitive module outputs before use in production
- Implement appropriate safeguards for autonomous operation
- Monitor system behavior for unexpected patterns
- Follow responsible AI development practices

### Dependencies

- Keep dependencies updated to address known vulnerabilities
- Review third-party integrations (mem0, LLM providers) for security
- Use virtual environments to isolate dependencies

### Data Handling

- Avoid storing sensitive data in AtomSpace without encryption
- Configure LLM API keys securely using environment variables
- Review memory persistence settings for data retention policies
