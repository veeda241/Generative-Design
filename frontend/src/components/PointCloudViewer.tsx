import { useRef, useMemo } from 'react';
import * as THREE from 'three';

interface PointCloudViewerProps {
    points: Array<{ pos?: [number, number, number]; color?: [number, number, number]; x?: number; y?: number; z?: number; r?: number; g?: number; b?: number }>;
    quality?: 'fast' | 'normal' | 'high' | 'ultra';
}

// Simple vertex shader
const vertexShader = `
    attribute vec3 color;
    varying vec3 vColor;
    uniform float pointSize;
    
    void main() {
        vColor = color;
        vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
        gl_PointSize = pointSize * (200.0 / max(-mvPosition.z, 0.1));
        gl_Position = projectionMatrix * mvPosition;
    }
`;

// Simple fragment shader - clean dots
const fragmentShader = `
    varying vec3 vColor;
    
    void main() {
        vec2 center = gl_PointCoord - vec2(0.5);
        float dist = length(center);
        if (dist > 0.5) discard;
        
        // Simple soft edge
        float alpha = 1.0 - smoothstep(0.3, 0.5, dist);
        gl_FragColor = vec4(vColor, alpha);
    }
`;

export const PointCloudViewer = ({ points, quality = 'high' }: PointCloudViewerProps) => {
    const groupRef = useRef<THREE.Group>(null);

    // Point size based on quality and count
    const pointSize = useMemo(() => {
        const count = points.length;
        const qualityMultiplier = {
            'fast': 2.0,
            'normal': 1.5,
            'high': 1.2,
            'ultra': 1.0
        }[quality] || 1.5;

        let baseSize = 0.02;
        if (count > 25000) baseSize = 0.008;
        else if (count > 15000) baseSize = 0.01;
        else if (count > 10000) baseSize = 0.012;
        else if (count > 5000) baseSize = 0.015;

        return baseSize * qualityMultiplier;
    }, [points.length, quality]);

    // Extract positions and colors, center and scale the model
    const { positions, colors } = useMemo(() => {
        const pos = new Float32Array(points.length * 3);
        const col = new Float32Array(points.length * 3);

        let minX = Infinity, minY = Infinity, minZ = Infinity;
        let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
        let sumX = 0, sumY = 0, sumZ = 0;

        // First pass: find bounds and center
        points.forEach((p) => {
            let x, y, z;
            if (p.pos) {
                x = p.pos[0]; y = p.pos[1]; z = p.pos[2];
            } else {
                x = p.x || 0; y = p.y || 0; z = p.z || 0;
            }
            minX = Math.min(minX, x); maxX = Math.max(maxX, x);
            minY = Math.min(minY, y); maxY = Math.max(maxY, y);
            minZ = Math.min(minZ, z); maxZ = Math.max(maxZ, z);
            sumX += x; sumY += y; sumZ += z;
        });

        const n = points.length || 1;
        const centerX = sumX / n;
        const centerY = sumY / n;
        const centerZ = sumZ / n;

        // Calculate scale to fit in a 2-unit cube
        const rangeX = maxX - minX || 1;
        const rangeY = maxY - minY || 1;
        const rangeZ = maxZ - minZ || 1;
        const maxRange = Math.max(rangeX, rangeY, rangeZ);
        const scaleFactor = 2.0 / maxRange;

        // Second pass: center, scale, and store
        points.forEach((p, i) => {
            let x, y, z;
            if (p.pos) {
                x = p.pos[0]; y = p.pos[1]; z = p.pos[2];
            } else {
                x = p.x || 0; y = p.y || 0; z = p.z || 0;
            }

            // Center and scale
            pos[i * 3] = (x - centerX) * scaleFactor;
            pos[i * 3 + 1] = (y - centerY) * scaleFactor;
            pos[i * 3 + 2] = (z - centerZ) * scaleFactor;

            if (p.color) {
                col[i * 3] = p.color[0];
                col[i * 3 + 1] = p.color[1];
                col[i * 3 + 2] = p.color[2];
            } else {
                col[i * 3] = p.r || 0.7;
                col[i * 3 + 1] = p.g || 0.75;
                col[i * 3 + 2] = p.b || 0.85;
            }
        });

        return {
            positions: pos,
            colors: col,
            scale: scaleFactor
        };
    }, [points]);

    // Shader material
    const shaderMaterial = useMemo(() => {
        return new THREE.ShaderMaterial({
            uniforms: { pointSize: { value: pointSize * 30 } },
            vertexShader,
            fragmentShader,
            transparent: true,
            depthWrite: true,
        });
    }, [pointSize]);

    // No auto-rotation - user controls with mouse

    return (
        <group ref={groupRef} position={[0, 0, 0]} rotation={[-Math.PI / 2, 0, 0]}>
            <points material={shaderMaterial}>
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
        </group>
    );
};
