import React, { useState, useEffect, useRef } from 'react';

const KawaiiHexapodSystem4 = () => {
  const [currentStep, setCurrentStep] = useState(0);
  const [hexapodPosition, setHexapodPosition] = useState({ x: 250, y: 300, rotation: 0 });
  const [obstacles, setObstacles] = useState([
    { x: 400, y: 250, size: 40 },
    { x: 350, y: 350, size: 30 }
  ]);
  const [isRunning, setIsRunning] = useState(false);
  const [showSystem4Details, setShowSystem4Details] = useState(true);
  const [emotionalState, setEmotionalState] = useState('curious');
  
  const canvasRef = useRef(null);
  
  // System 4 term sequence
  const termSequence = [1, 4, 2, 8, 5, 7];
  
  // Three sets running out of phase
  const getSetsAtStep = (step) => {
    const cycle = step % 12;
    const setConfigs = [
      // Set 1, Set 2, Set 3
      [
        { term: 8, mode: 'E' },
        { term: 7, mode: 'R' },
        { term: 4, mode: 'E' }
      ],
      [
        { term: 5, mode: 'E' },
        { term: 1, mode: 'R' },
        { term: 2, mode: 'E' }
      ],
      [
        { term: 7, mode: 'E' },
        { term: 4, mode: 'R' },
        { term: 8, mode: 'E' }
      ],
      [
        { term: 1, mode: 'E' },
        { term: 2, mode: 'R' },
        { term: 5, mode: 'R' }
      ],
      [
        { term: 4, mode: 'E' },
        { term: 8, mode: 'E' },
        { term: 7, mode: 'R' }
      ],
      [
        { term: 2, mode: 'E' },
        { term: 5, mode: 'E' },
        { term: 1, mode: 'R' }
      ],
      [
        { term: 8, mode: 'E' },
        { term: 7, mode: 'E' },
        { term: 4, mode: 'R' }
      ],
      [
        { term: 5, mode: 'R' },
        { term: 1, mode: 'E' },
        { term: 2, mode: 'R' }
      ],
      [
        { term: 7, mode: 'R' },
        { term: 4, mode: 'E' },
        { term: 8, mode: 'E' }
      ],
      [
        { term: 1, mode: 'R' },
        { term: 2, mode: 'E' },
        { term: 5, mode: 'E' }
      ],
      [
        { term: 4, mode: 'R' },
        { term: 8, mode: 'E' },
        { term: 7, mode: 'E' }
      ],
      [
        { term: 2, mode: 'R' },
        { term: 5, mode: 'R' },
        { term: 1, mode: 'E' }
      ]
    ];
    
    return setConfigs[cycle];
  };
  
  // Term meanings for the hexapod
  const termMeanings = {
    1: 'Perception (Need vs Capacity)',
    2: 'Idea Formation',
    4: 'Sensory Input',
    5: 'Action Sequence',
    7: 'Memory Encoding',
    8: 'Balanced Response'
  };
  
  // Process one step of System 4
  const processStep = () => {
    const sets = getSetsAtStep(currentStep);
    
    // Check for tensional coupling between T4E and T7R
    const hasT4E = sets.some(s => s.term === 4 && s.mode === 'E');
    const hasT7R = sets.some(s => s.term === 7 && s.mode === 'R');
    
    if (hasT4E && hasT7R) {
      // Memory-guided perception - check for obstacles
      const nearestObstacle = findNearestObstacle();
      if (nearestObstacle && nearestObstacle.distance < 100) {
        setEmotionalState('concerned');
        planAvoidanceRoute(nearestObstacle);
      } else {
        setEmotionalState('happy');
      }
    }
    
    // Execute movement based on active terms
    if (sets.some(s => s.term === 8 && s.mode === 'E')) {
      // Balanced motor response
      executeMovement();
    }
    
    if (sets.some(s => s.term === 5)) {
      // Action sequence - animate legs
      animateLegs();
    }
  };
  
  const findNearestObstacle = () => {
    let nearest = null;
    let minDist = Infinity;
    
    obstacles.forEach(obs => {
      const dist = Math.sqrt(
        Math.pow(obs.x - hexapodPosition.x, 2) + 
        Math.pow(obs.y - hexapodPosition.y, 2)
      );
      if (dist < minDist) {
        minDist = dist;
        nearest = { ...obs, distance: dist };
      }
    });
    
    return nearest;
  };
  
  const planAvoidanceRoute = (obstacle) => {
    // Simple avoidance - turn away from obstacle
    const angleToObstacle = Math.atan2(
      obstacle.y - hexapodPosition.y,
      obstacle.x - hexapodPosition.x
    );
    const currentAngle = hexapodPosition.rotation * Math.PI / 180;
    const angleDiff = angleToObstacle - currentAngle;
    
    // Turn away
    const newRotation = hexapodPosition.rotation - (angleDiff * 180 / Math.PI) * 0.3;
    setHexapodPosition(prev => ({ ...prev, rotation: newRotation }));
  };
  
  const executeMovement = () => {
    const speed = 2;
    const angle = hexapodPosition.rotation * Math.PI / 180;
    
    setHexapodPosition(prev => ({
      ...prev,
      x: prev.x + Math.cos(angle) * speed,
      y: prev.y + Math.sin(angle) * speed,
      // Wrap around canvas
      x: ((prev.x + Math.cos(angle) * speed) + 600) % 600,
      y: ((prev.y + Math.sin(angle) * speed) + 400) % 400
    }));
  };
  
  const animateLegs = () => {
    // Leg animation handled in drawing
  };
  
  // Main animation loop
  useEffect(() => {
    let interval;
    if (isRunning) {
      interval = setInterval(() => {
        setCurrentStep(step => step + 1);
        processStep();
      }, 500);
    }
    return () => clearInterval(interval);
  }, [isRunning, currentStep, hexapodPosition]);
  
  // Drawing
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, 600, 400);
    
    // Draw obstacles
    obstacles.forEach(obs => {
      ctx.fillStyle = '#FFB6C1';
      ctx.beginPath();
      ctx.arc(obs.x, obs.y, obs.size, 0, 2 * Math.PI);
      ctx.fill();
      ctx.strokeStyle = '#FF69B4';
      ctx.lineWidth = 2;
      ctx.stroke();
    });
    
    // Draw hexapod
    ctx.save();
    ctx.translate(hexapodPosition.x, hexapodPosition.y);
    ctx.rotate(hexapodPosition.rotation * Math.PI / 180);
    
    // Body (cute oval)
    ctx.fillStyle = '#9370DB';
    ctx.beginPath();
    ctx.ellipse(0, 0, 30, 20, 0, 0, 2 * Math.PI);
    ctx.fill();
    ctx.strokeStyle = '#4B0082';
    ctx.lineWidth = 2;
    ctx.stroke();
    
    // Draw 6 legs in 3 pairs
    const legPhase = (currentStep % 6) * Math.PI / 3;
    const sets = getSetsAtStep(currentStep);
    
    for (let pair = 0; pair < 3; pair++) {
      const pairOffset = pair * 15 - 15;
      const setData = sets[pair];
      const isActive = setData.term === 5 || setData.term === 8;
      
      // Left leg
      ctx.strokeStyle = isActive ? '#FF1493' : '#DDA0DD';
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(pairOffset, -20);
      const leftAngle = -Math.PI/3 + Math.sin(legPhase + pair * 2) * 0.3;
      ctx.lineTo(
        pairOffset + Math.cos(leftAngle) * 25,
        -20 + Math.sin(leftAngle) * 25
      );
      ctx.stroke();
      
      // Right leg
      ctx.beginPath();
      ctx.moveTo(pairOffset, 20);
      const rightAngle = Math.PI/3 + Math.sin(legPhase + pair * 2 + Math.PI) * 0.3;
      ctx.lineTo(
        pairOffset + Math.cos(rightAngle) * 25,
        20 + Math.sin(rightAngle) * 25
      );
      ctx.stroke();
    }
    
    // Cute face
    const eyeSize = emotionalState === 'happy' ? 5 : 
                    emotionalState === 'concerned' ? 7 : 6;
    
    // Eyes
    ctx.fillStyle = '#000';
    ctx.beginPath();
    ctx.arc(-8, -5, eyeSize, 0, 2 * Math.PI);
    ctx.fill();
    ctx.beginPath();
    ctx.arc(8, -5, eyeSize, 0, 2 * Math.PI);
    ctx.fill();
    
    // Sparkles in eyes
    ctx.fillStyle = '#FFF';
    ctx.beginPath();
    ctx.arc(-6, -7, 2, 0, 2 * Math.PI);
    ctx.fill();
    ctx.beginPath();
    ctx.arc(10, -7, 2, 0, 2 * Math.PI);
    ctx.fill();
    
    // Mouth
    ctx.strokeStyle = '#000';
    ctx.lineWidth = 2;
    ctx.beginPath();
    if (emotionalState === 'happy') {
      ctx.arc(0, 0, 8, 0.2 * Math.PI, 0.8 * Math.PI);
    } else if (emotionalState === 'concerned') {
      ctx.arc(0, 8, 8, 1.2 * Math.PI, 1.8 * Math.PI);
    } else {
      ctx.moveTo(-5, 5);
      ctx.lineTo(5, 5);
    }
    ctx.stroke();
    
    // Antennae showing current dominant term
    const dominantSet = sets.find(s => s.mode === 'E' && [4, 8].includes(s.term)) || sets[0];
    ctx.strokeStyle = dominantSet.mode === 'E' ? '#FF69B4' : '#87CEEB';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(-5, -20);
    ctx.lineTo(-8, -35);
    ctx.moveTo(5, -20);
    ctx.lineTo(8, -35);
    ctx.stroke();
    
    // Antenna tips glow based on term
    const glowColor = {
      1: '#FFD700', 2: '#FF69B4', 4: '#00CED1',
      5: '#32CD32', 7: '#9370DB', 8: '#FF6347'
    }[dominantSet.term];
    
    ctx.fillStyle = glowColor;
    ctx.beginPath();
    ctx.arc(-8, -35, 4, 0, 2 * Math.PI);
    ctx.arc(8, -35, 4, 0, 2 * Math.PI);
    ctx.fill();
    
    ctx.restore();
    
  }, [currentStep, hexapodPosition, obstacles, emotionalState]);

  return (
    <div className="p-6 max-w-6xl mx-auto">
      <h2 className="text-2xl font-bold mb-4">✨ Kawaii Hexapod: Living System 4 ✨</h2>
      
      <div className="mb-4 p-4 bg-purple-50 rounded">
        <p className="text-sm">
          Watch as our adorable hexapod friend navigates using Campbell's System 4! 
          Three sets of consciousness run simultaneously, creating intelligent behavior 
          from the interplay of Expressive (reactive) and Regenerative (anticipatory) modes.
        </p>
      </div>
      
      <div className="grid grid-cols-2 gap-6">
        <div>
          <canvas 
            ref={canvasRef} 
            width={600} 
            height={400} 
            className="border-2 border-purple-300 rounded-lg bg-purple-50"
          />
          
          <div className="mt-4 flex gap-2">
            <button
              onClick={() => setIsRunning(!isRunning)}
              className={`px-4 py-2 rounded font-semibold transition-colors ${
                isRunning 
                  ? 'bg-red-400 hover:bg-red-500 text-white' 
                  : 'bg-green-400 hover:bg-green-500 text-white'
              }`}
            >
              {isRunning ? '⏸️ Pause' : '▶️ Start'}
            </button>
            
            <button
              onClick={() => {
                setCurrentStep(0);
                setHexapodPosition({ x: 250, y: 300, rotation: 0 });
                setEmotionalState('curious');
              }}
              className="px-4 py-2 bg-blue-400 hover:bg-blue-500 text-white rounded font-semibold"
            >
              🔄 Reset
            </button>
            
            <button
              onClick={() => {
                const newObs = {
                  x: Math.random() * 500 + 50,
                  y: Math.random() * 300 + 50,
                  size: Math.random() * 20 + 20
                };
                setObstacles([...obstacles, newObs]);
              }}
              className="px-4 py-2 bg-pink-400 hover:bg-pink-500 text-white rounded font-semibold"
            >
              ➕ Add Obstacle
            </button>
          </div>
        </div>
        
        <div>
          <h3 className="font-bold mb-2">System 4 Status - Step {currentStep}</h3>
          
          <div className="space-y-3">
            {getSetsAtStep(currentStep).map((set, idx) => (
              <div 
                key={idx}
                className={`p-3 rounded border-2 ${
                  set.mode === 'E' 
                    ? 'bg-pink-50 border-pink-300' 
                    : 'bg-blue-50 border-blue-300'
                }`}
              >
                <div className="flex justify-between items-center">
                  <span className="font-semibold">Set {idx + 1}:</span>
                  <span className={`px-2 py-1 rounded text-xs font-bold ${
                    set.mode === 'E' 
                      ? 'bg-pink-200 text-pink-800' 
                      : 'bg-blue-200 text-blue-800'
                  }`}>
                    {set.mode === 'E' ? 'Expressive' : 'Regenerative'}
                  </span>
                </div>
                <div className="text-sm mt-1">
                  <strong>T{set.term}:</strong> {termMeanings[set.term]}
                </div>
              </div>
            ))}
          </div>
          
          {showSystem4Details && (
            <div className="mt-4 p-3 bg-yellow-50 rounded">
              <h4 className="font-semibold mb-2">🧠 Active Couplings:</h4>
              <div className="text-sm space-y-1">
                {getSetsAtStep(currentStep).some(s => s.term === 4 && s.mode === 'E') &&
                 getSetsAtStep(currentStep).some(s => s.term === 7 && s.mode === 'R') && (
                  <div className="text-green-700">
                    ✓ T4E ↔ T7R: Sensory input coupled with memory recall
                  </div>
                )}
                {getSetsAtStep(currentStep).some(s => s.term === 1 && s.mode === 'R') &&
                 getSetsAtStep(currentStep).some(s => s.term === 2 && s.mode === 'E') && (
                  <div className="text-blue-700">
                    ✓ T1R ↔ T2E: Simulation reconciling with reactive planning
                  </div>
                )}
                {getSetsAtStep(currentStep).filter(s => s.term === 8 && s.mode === 'E').length > 0 && (
                  <div className="text-purple-700">
                    ✓ T8E: Executing balanced motor response
                  </div>
                )}
              </div>
            </div>
          )}
          
          <div className="mt-4 p-3 bg-purple-100 rounded">
            <h4 className="font-semibold mb-1">Emotional State: {emotionalState}</h4>
            <div className="text-sm">
              {emotionalState === 'happy' && "No obstacles detected - exploring freely! 😊"}
              {emotionalState === 'concerned' && "Obstacle nearby - planning avoidance route! 😟"}
              {emotionalState === 'curious' && "Scanning environment... 🤔"}
            </div>
          </div>
        </div>
      </div>
      
      <div className="mt-6 p-4 bg-gray-50 rounded">
        <h3 className="font-semibold mb-2">The Magic of Three Sets</h3>
        <p className="text-sm">
          Our kawaii friend runs three parallel streams of consciousness, each one step out of 
          phase. This creates the beautiful braiding of past experience (memory), present 
          sensation (input), and future planning (simulation) that enables truly intelligent 
          navigation. The pink obstacles trigger memory-guided avoidance behaviors when 
          T4E (sensory) couples with T7R (memory recall)!
        </p>
      </div>
    </div>
  );
};

export default KawaiiHexapodSystem4;