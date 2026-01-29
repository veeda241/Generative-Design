import { useRef, useMemo, useState } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';

interface PointCloudViewerProps {
    points: Array<{ pos?: [number, number, number]; color?: [number, number, number]; x?: number; y?: number; z?: number; r?: number; g?: number; b?: number }>;
    quality?: 'fast' | 'normal' | 'high' | 'ultra';
}

// Custom shader for higher quality point rendering
const vertexShader = `
    attribute vec3 color;
    varying vec3 vColor;
    varying float vDistance;
    uniform float pointSize;
    
    void main() {
        vColor = color;
        vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
        vDistance = -mvPosition.z;
        gl_PointSize = pointSize * (300.0 / vDistance);
        gl_Position = projectionMatrix * mvPosition;
    }
`;

const fragmentShader = `
    varying vec3 vColor;
    varying float vDistance;
    
    void main() {
        // Create smooth circular points instead of squares
        vec2 center = gl_PointCoord - vec2(0.5);
        float dist = length(center);
        
        // Smooth anti-aliased edge
        float alpha = 1.0 - smoothstep(0.35, 0.5, dist);
        
        if (alpha < 0.01) discard;
        
        // Add subtle shading for 3D appearance
        float shade = 1.0 - dist * 0.5;
        vec3 shadedColor = vColor * shade;
        
        // Add slight ambient occlusion effect
        float ao = 1.0 - (dist * 0.3);
        
        gl_FragColor = vec4(shadedColor * ao, alpha);
    }
`;

export const PointCloudViewer = ({ points, quality = 'high' }: PointCloudViewerProps) => {
    const pointsRef = useRef<THREE.Points>(null);
    const [useShader] = useState(quality === 'high' || quality === 'ultra');

    // Dynamic point size based on point count and quality
    const pointSize = useMemo(() => {
        const baseSize = {
            'fast': 0.08,
            'normal': 0.06,
            'high': 0.045,
            'ultra': 0.035
        }[quality] || 0.05;
        
        // Adjust size based on point density
        if (points.length > 10000) return baseSize * 0.7;
        if (points.length > 5000) return baseSize * 0.85;
        return baseSize;
    }, [points.length, quality]);

    const { positions, colors } = useMemo(() => {
        const pos = new Float32Array(points.length * 3);
        const col = new Float32Array(points.length * 3);

        points.forEach((p, i) => {
            // Handle both formats: {pos, color} and {x, y, z, r, g, b}
            if (p.pos) {
                pos[i * 3] = p.pos[0];
                pos[i * 3 + 1] = p.pos[1];
                pos[i * 3 + 2] = p.pos[2];
            } else {
                pos[i * 3] = p.x || 0;
                pos[i * 3 + 1] = p.y || 0;
                pos[i * 3 + 2] = p.z || 0;
            }

            if (p.color) {
                // Enhance color saturation for better visibility
                const r = Math.min(1.0, p.color[0] * 1.1);
                const g = Math.min(1.0, p.color[1] * 1.1);
                const b = Math.min(1.0, p.color[2] * 1.1);
                col[i * 3] = r;
                col[i * 3 + 1] = g;
                col[i * 3 + 2] = b;
            } else {
                col[i * 3] = p.r || 0.5;
                col[i * 3 + 1] = p.g || 0.5;
                col[i * 3 + 2] = p.b || 0.5;
            }
        });

        return { positions: pos, colors: col };
    }, [points]);

    // Custom shader material for high quality rendering
    const shaderMaterial = useMemo(() => {
        return new THREE.ShaderMaterial({
            uniforms: {
                pointSize: { value: pointSize * 100 }
            },
            vertexShader,
            fragmentShader,
            transparent: true,
            depthWrite: true,
            blending: THREE.NormalBlending
        });
    }, [pointSize]);

    useFrame(() => {
        if (pointsRef.current) {
            pointsRef.current.rotation.y += 0.002;
        }
    });

    // Use custom shader for high/ultra quality, standard material for fast/normal
    if (useShader) {
        return (
            <points ref={pointsRef} material={shaderMaterial}>
                <bufferGeometry>
                    <bufferAttribute
                        attach="attributes-position"
                        count={positions.length / 3}
                        array={positions}
                        itemSize={3}
                        args={[positions, 3]}
                    />
                    <bufferAttribute
                        attach="attributes-color"
                        count={colors.length / 3}
                        array={colors}
                        itemSize={3}
                        args={[colors, 3]}
                    />
                </bufferGeometry>
            </points>
        );
    }

    return (
        <points ref={pointsRef}>
            <bufferGeometry>
                <bufferAttribute
                    attach="attributes-position"
                    count={positions.length / 3}
                    array={positions}
                    itemSize={3}
                    args={[positions, 3]}
                />
                <bufferAttribute
                    attach="attributes-color"
                    count={colors.length / 3}
                    array={colors}
                    itemSize={3}
                    args={[colors, 3]}
                />
            </bufferGeometry>
            <pointsMaterial
                size={pointSize}
                vertexColors
                transparent
                opacity={0.95}
                sizeAttenuation={true}
            />
        </points>
    );
};
