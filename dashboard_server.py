#!/usr/bin/env python3
"""
CogPrime AGI Visualization Dashboard Server
Real-time cognitive state monitoring with WebSocket support
"""

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path

try:
    from aiohttp import web
    import aiohttp_cors
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    print("⚠️  aiohttp not installed. Install with: pip install aiohttp aiohttp-cors")

import sys
sys.path.insert(0, str(Path(__file__).parent))

try:
    from src.core.cognitive_core import CogPrimeCore, CognitiveState
    from src.modules.perception import SensoryInput
    import torch
    COGPRIME_AVAILABLE = True
except ImportError as e:
    COGPRIME_AVAILABLE = False
    print(f"⚠️  CogPrime modules not fully available: {e}")
    print("Running in demo mode with simulated data")


class CogPrimeDashboard:
    """Real-time AGI consciousness monitoring dashboard"""
    
    def __init__(self):
        self.cognitive_system = None
        self.metrics_history = []
        self.websockets = set()
        self.running = False
        
        if COGPRIME_AVAILABLE:
            try:
                config = {
                    'visual_dim': 784,
                    'audio_dim': 256,
                    'memory_size': 1000
                }
                self.cognitive_system = CogPrimeCore(config)
                print("✅ CogPrime cognitive system initialized")
            except Exception as e:
                print(f"⚠️  Could not initialize CogPrime: {e}")
                print("Running in demo mode")
    
    def get_current_metrics(self):
        """Get current cognitive metrics"""
        if self.cognitive_system and COGPRIME_AVAILABLE:
            try:
                # Run a cognitive cycle with random input
                sensory_input = SensoryInput(
                    visual=torch.randn(784),
                    auditory=torch.randn(256)
                )
                action = self.cognitive_system.cognitive_cycle(sensory_input, reward=1.0)
                
                # Extract real metrics from the system
                state = self.cognitive_system.state
                
                metrics = {
                    'timestamp': datetime.now().isoformat(),
                    'consciousness_level': float(torch.sigmoid(state.attention_focus.mean()).item()),
                    'relevance_realization': float(state.emotional_valence),
                    'attention_entropy': float(torch.distributions.Categorical(
                        probs=torch.softmax(state.attention_focus[:10], dim=0)
                    ).entropy().item()),
                    'pattern_diversity': len(state.working_memory) / 100.0,
                    'meta_learning_rate': 0.001 + abs(float(state.last_reward)) * 0.001,
                    'cycle_count': len(self.metrics_history) + 1,
                    'goal_stack_depth': len(state.goal_stack),
                    'last_action': str(action),
                    'total_reward': float(state.total_reward)
                }
            except Exception as e:
                print(f"Error getting real metrics: {e}")
                metrics = self._get_demo_metrics()
        else:
            metrics = self._get_demo_metrics()
        
        self.metrics_history.append(metrics)
        if len(self.metrics_history) > 1000:
            self.metrics_history = self.metrics_history[-1000:]
        
        return metrics
    
    def _get_demo_metrics(self):
        """Generate demo metrics for visualization"""
        import random
        import math
        
        t = time.time()
        return {
            'timestamp': datetime.now().isoformat(),
            'consciousness_level': 0.5 + 0.3 * math.sin(t * 0.5),
            'relevance_realization': 0.7 + 0.2 * math.cos(t * 0.3),
            'attention_entropy': 4.0 + random.random() * 2.0,
            'pattern_diversity': 0.15 + random.random() * 0.05,
            'meta_learning_rate': 0.001 + random.random() * 0.005,
            'cycle_count': len(self.metrics_history) + 1,
            'goal_stack_depth': random.randint(1, 5),
            'last_action': random.choice(['focus_attention', 'query_memory', 'explore', 'exploit']),
            'total_reward': len(self.metrics_history) * 0.1 + random.random()
        }
    
    def get_evolution_data(self):
        """Get evolutionary progress data"""
        if len(self.metrics_history) < 2:
            return {'generations': [], 'fitness': [], 'diversity': []}
        
        recent = self.metrics_history[-50:]
        return {
            'generations': list(range(len(recent))),
            'fitness': [m['consciousness_level'] * 100 for m in recent],
            'diversity': [m['pattern_diversity'] * 500 for m in recent]
        }
    
    def get_attention_allocation(self):
        """Get attention allocation data"""
        return {
            'labels': ['Perception', 'Reasoning', 'Action', 'Learning', 'Meta-Cognition'],
            'values': [35, 25, 20, 15, 5]
        }
    
    async def broadcast_metrics(self):
        """Broadcast metrics to all connected WebSocket clients"""
        while self.running:
            try:
                metrics = self.get_current_metrics()
                message = json.dumps({
                    'type': 'metrics_update',
                    'data': metrics
                })
                
                # Send to all connected websockets
                for ws in list(self.websockets):
                    try:
                        await ws.send_str(message)
                    except Exception as e:
                        print(f"Error sending to websocket: {e}")
                        self.websockets.discard(ws)
                
                await asyncio.sleep(1)  # Update every second
            except Exception as e:
                print(f"Error in broadcast_metrics: {e}")
                await asyncio.sleep(1)


async def websocket_handler(request):
    """Handle WebSocket connections"""
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    
    dashboard = request.app['dashboard']
    dashboard.websockets.add(ws)
    
    print(f"✅ WebSocket client connected (total: {len(dashboard.websockets)})")
    
    try:
        async for msg in ws:
            if msg.type == web.WSMsgType.TEXT:
                data = json.loads(msg.data)
                
                if data.get('type') == 'get_evolution':
                    evolution_data = dashboard.get_evolution_data()
                    await ws.send_str(json.dumps({
                        'type': 'evolution_data',
                        'data': evolution_data
                    }))
                
                elif data.get('type') == 'get_attention':
                    attention_data = dashboard.get_attention_allocation()
                    await ws.send_str(json.dumps({
                        'type': 'attention_data',
                        'data': attention_data
                    }))
            
            elif msg.type == web.WSMsgType.ERROR:
                print(f'WebSocket error: {ws.exception()}')
    
    finally:
        dashboard.websockets.discard(ws)
        print(f"❌ WebSocket client disconnected (remaining: {len(dashboard.websockets)})")
    
    return ws


async def index_handler(request):
    """Serve the dashboard HTML"""
    html_path = Path(__file__).parent / 'agi_visualization_dashboard.html'
    if html_path.exists():
        return web.FileResponse(html_path)
    else:
        return web.Response(text="Dashboard HTML not found", status=404)


async def metrics_handler(request):
    """REST API endpoint for metrics"""
    dashboard = request.app['dashboard']
    metrics = dashboard.get_current_metrics()
    return web.json_response(metrics)


async def start_background_tasks(app):
    """Start background tasks"""
    dashboard = app['dashboard']
    dashboard.running = True
    app['metrics_task'] = asyncio.create_task(dashboard.broadcast_metrics())


async def cleanup_background_tasks(app):
    """Cleanup background tasks"""
    dashboard = app['dashboard']
    dashboard.running = False
    app['metrics_task'].cancel()
    await app['metrics_task']


def create_app():
    """Create and configure the web application"""
    app = web.Application()
    
    # Initialize dashboard
    dashboard = CogPrimeDashboard()
    app['dashboard'] = dashboard
    
    # Configure CORS
    cors = aiohttp_cors.setup(app, defaults={
        "*": aiohttp_cors.ResourceOptions(
            allow_credentials=True,
            expose_headers="*",
            allow_headers="*",
        )
    })
    
    # Add routes
    app.router.add_get('/', index_handler)
    app.router.add_get('/ws', websocket_handler)
    app.router.add_get('/api/metrics', metrics_handler)
    
    # Configure CORS on all routes
    for route in list(app.router.routes()):
        cors.add(route)
    
    # Setup background tasks
    app.on_startup.append(start_background_tasks)
    app.on_cleanup.append(cleanup_background_tasks)
    
    return app


def main():
    """Main entry point"""
    if not AIOHTTP_AVAILABLE:
        print("\n❌ Cannot start server: aiohttp not installed")
        print("Install with: pip install aiohttp aiohttp-cors")
        return
    
    print("=" * 60)
    print("🧠 CogPrime AGI Visualization Dashboard Server")
    print("=" * 60)
    print()
    
    if COGPRIME_AVAILABLE:
        print("✅ Running with real CogPrime cognitive system")
    else:
        print("⚠️  Running in demo mode with simulated data")
    
    print()
    print("Starting server on http://localhost:8080")
    print("Open your browser and navigate to: http://localhost:8080")
    print()
    print("Press Ctrl+C to stop the server")
    print("=" * 60)
    print()
    
    app = create_app()
    web.run_app(app, host='0.0.0.0', port=8080)


if __name__ == '__main__':
    main()
