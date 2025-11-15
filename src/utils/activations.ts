export function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

export const activations = {
  relu: { name: "ReLU", f: (z: number) => (z > 0 ? z : 0), df: (z: number) => (z > 0 ? 1 : 0) },
  tanh: { name: "tanh", f: (z: number) => Math.tanh(z), df: (z: number) => 1 - Math.tanh(z) ** 2 },
  sigmoid: {
    name: "Sigmoid",
    f: (z: number) => sigmoid(z),
    df: (z: number) => {
      const s = sigmoid(z);
      return s * (1 - s);
    },
  },
  silu: {
    name: "SiLU",
    f: (z: number) => z * sigmoid(z),
    df: (z: number) => {
      const s = sigmoid(z);
      return s + z * s * (1 - s);
    },
  },
  gelu: {
    name: "GELU",
    f: (z: number) => {
      const c = Math.sqrt(2 / Math.PI);
      const u = c * (z + 0.044715 * z * z * z);
      return 0.5 * z * (1 + Math.tanh(u));
    },
    df: (z: number) => {
      const c = Math.sqrt(2 / Math.PI);
      const zz = z * z;
      const u = c * (z + 0.044715 * z * zz);
      const th = Math.tanh(u);
      const sech2 = 1 - th * th;
      const du = c * (1 + 0.134145 * zz);
      return 0.5 * (1 + th) + 0.5 * z * sech2 * du;
    },
  },
} as const;

export type ActKey = keyof typeof activations;
