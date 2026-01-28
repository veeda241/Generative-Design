import { useRef, useMemo } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';

interface PointCloudViewerProps {
    points: Array<{ pos?: [number, number, number]; color?: [number, number, number]; x?: number; y?: number; z?: number; r?: number; g?: number; b?: number }>;
}

export const PointCloudViewer = ({ points }: PointCloudViewerProps) => {
    const pointsRef = useRef<THREE.Points>(null);

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
                col[i * 3] = p.color[0];
                col[i * 3 + 1] = p.color[1];
                col[i * 3 + 2] = p.color[2];
            } else {
                col[i * 3] = p.r || 0.5;
                col[i * 3 + 1] = p.g || 0.5;
                col[i * 3 + 2] = p.b || 0.5;
            }
        });

        return { positions: pos, colors: col };
    }, [points]);

    useFrame(() => {
        if (pointsRef.current) {
            pointsRef.current.rotation.y += 0.002;
        }
    });

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
                size={0.15}
                vertexColors
                transparent
                opacity={0.9}
                sizeAttenuation={true}
            />
        </points>
    );
};
