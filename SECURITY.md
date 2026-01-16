# Security Policy

## Security Measures Implemented

### Container Security
- **Non-root user**: Application runs as unprivileged user `youlama` (UID 1000)
- **Dropped capabilities**: All Linux capabilities are dropped
- **No new privileges**: Prevents privilege escalation
- **Resource limits**: Memory limited to 32GB to prevent DoS
- **Localhost binding**: Service only accessible on 127.0.0.1

### Application Security
- **Gradio share disabled**: Prevents public exposure through relay servers
- **Analytics disabled**: No telemetry data sent to Gradio servers
- **Input validation**: YouTube URL validation before processing
- **Temp file cleanup**: Automatic cleanup of downloaded audio files

### Network Security
- **No public exposure**: Service binds to localhost only
- **Use reverse proxy**: For external access, use nginx/traefik with TLS
- **Isolated network**: Docker bridge network with IP masquerade

## Known Limitations

1. **No authentication**: Gradio interface has no built-in auth
   - Mitigation: Use reverse proxy with authentication
   
2. **Subtitle temp files**: May not be cleaned up in all cases
   - Mitigation: Temp volume can be periodically cleaned

3. **URL logging**: YouTube URLs are logged
   - Mitigation: Configure log rotation, don't expose logs

## Recommended Production Setup

```yaml
# Use with reverse proxy (e.g., Traefik)
services:
  traefik:
    image: traefik:v2.10
    command:
      - "--providers.docker=true"
      - "--entrypoints.websecure.address=:443"
    ports:
      - "443:443"
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock:ro
      
  youlama:
    # ... existing config ...
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.youlama.rule=Host(`youlama.yourdomain.com`)"
      - "traefik.http.routers.youlama.tls=true"
```

## Reporting Security Issues

If you discover a security vulnerability, please report it privately.
Do not create public GitHub issues for security vulnerabilities.

## Security Checklist

- [ ] Use secure Docker Compose file (`docker-compose.secure.yml`)
- [ ] Keep `share = false` in config.ini
- [ ] Use reverse proxy with TLS for external access
- [ ] Enable authentication at proxy level
- [ ] Regularly update base images
- [ ] Monitor container logs for suspicious activity
- [ ] Use secrets management for any credentials
