import { useEffect, useRef, useMemo, useState } from 'react';
import * as THREE from 'three';

interface MeshViewerProps {
    url?: string;
    geometry?: THREE.BufferGeometry;
    color?: string;
}

// STL Loader class (inline to avoid import issues)
class STLLoader extends THREE.Loader {
    parse(data: ArrayBuffer): THREE.BufferGeometry {
        const geometry = new THREE.BufferGeometry();
        const dataView = new DataView(data);

        // Skip header (80 bytes)
        const numTriangles = dataView.getUint32(80, true);
        const vertices: number[] = [];
        const normals: number[] = [];

        let offset = 84;
        for (let i = 0; i < numTriangles; i++) {
            // Normal
            const nx = dataView.getFloat32(offset, true);
            const ny = dataView.getFloat32(offset + 4, true);
            const nz = dataView.getFloat32(offset + 8, true);
            offset += 12;

            // Vertices (3 per triangle)
            for (let j = 0; j < 3; j++) {
                const x = dataView.getFloat32(offset, true);
                const y = dataView.getFloat32(offset + 4, true);
                const z = dataView.getFloat32(offset + 8, true);
                offset += 12;

                vertices.push(x, y, z);
                normals.push(nx, ny, nz);
            }

            // Attribute byte count
            offset += 2;
        }

        geometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
        geometry.setAttribute('normal', new THREE.Float32BufferAttribute(normals, 3));

        return geometry;
    }
}

export const MeshViewer = ({ url, geometry: providedGeometry, color = '#38bdf8' }: MeshViewerProps) => {
    const meshRef = useRef<THREE.Mesh>(null);
    const [localGeometry, setLocalGeometry] = useState<THREE.BufferGeometry | null>(null);
    const [loadError, setLoadError] = useState<string | null>(null);

    // Load STL from URL if provided
    useEffect(() => {
        if (url) {
            fetch(url)
                .then(response => response.arrayBuffer())
                .then(data => {
                    const loader = new STLLoader();
                    const geometry = loader.parse(data);

                    // Center the geometry
                    geometry.center();
                    geometry.computeVertexNormals();

                    // Scale to fit in view
                    geometry.computeBoundingBox();
                    const box = geometry.boundingBox!;
                    const size = new THREE.Vector3();
                    box.getSize(size);
                    const maxDim = Math.max(size.x, size.y, size.z);
                    const scale = 2 / maxDim;
                    geometry.scale(scale, scale, scale);

                    setLocalGeometry(geometry);
                })
                .catch(err => {
                    console.error('Failed to load STL:', err);
                    setLoadError(err.message);
                });
        }
    }, [url]);

    // Use provided geometry or loaded geometry
    const geometry = providedGeometry || localGeometry;

    // Create material - Bright, visible colors
    const material = useMemo(() => {
        return new THREE.MeshStandardMaterial({
            color: color,
            metalness: 0.1,
            roughness: 0.6,
            flatShading: false,
            side: THREE.DoubleSide,
            emissive: color,
            emissiveIntensity: 0.15,
        });
    }, [color]);

    if (loadError) {
        console.error('Mesh load error:', loadError);
        return null;
    }

    if (!geometry) {
        return null;
    }

    return (
        <mesh ref={meshRef} geometry={geometry} material={material} rotation={[-Math.PI / 2, 0, 0]}>
        </mesh>
    );
};
