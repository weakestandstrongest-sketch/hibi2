// chat templates are imported as text (vite ?raw)
declare module "*.jinja?raw" {
  const text: string;
  export default text;
}
