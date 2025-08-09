## Guía para reconfigurar tu repositorio

En este curso vamos a usar una configuración novedosa de tu repositorio de entregas, que te da control total, mantiene tu trabajo privado y te permite sincronizar los materiales de la cátedra fácilmente.

Sigue estos pasos **una sola vez** con atención.

### Parte 1: Convertir tu fork en un repositorio privado.

Primero, vamos a desvincular tu repositorio de la red de forks para poder cambiar su visibilidad.

1.  Ve a la página principal de **tu fork** en GitHub.
2.  Haz clic en la pestaña **`Settings`**.
3.  Baja hasta el final, a la sección roja llamada **`Danger Zone`**.
4.  **Desvincular el Fork**: Haz clic en el botón **`Leave fork network`**.
      * GitHub te pedirá que leas unas advertencias y que confirmes escribiendo el nombre de tu repositorio.
5.  **Cambiar Visibilidad**: Una vez desvinculado, la opción **`Change visibility`** (la primera en la `Danger Zone`) estará activa.
      * Haz clic en ella y selecciona la opción para hacerlo **`Private`**.

### Parte 2: Agregar a los docentes como colaboradores.

Para que podamos ver tu trabajo y corregirlo, debes agregarnos como colaboradores.

1.  En la misma página de **`Settings`**, ve a la sección **`Collaborators`** en el menú de la izquierda.
2.  Haz clic en el botón verde **`Add people`**.
3.  Agrega a los siguientes usuarios (debes hacerlo uno por uno):
      * `glandfried`
      * `constanzadegalvagni`
4.  Ellos recibirán una invitación por correo que deberán aceptar para tener acceso.

### Parte 3: Reorganizar tus archivos y agregar el repositorio de la materia como un submódulo.

Agregar el respositorio de la materia como submodulo te va a permitir tener los materiales en una simple carpeta y mantenerlos actualizados a medida que avanza la cursada.
Esto te permite trabajar libremente en el resto de tu repositorio sin que se produzca conflicto alguno con la actualización del repositorio de la materia.
Para poder automatizar las correcciones, vamos a pedir que tus entregas estén adentro de una carpeta llamada `entregas`.
Ahora vamos a organizar la estructura de archivos de tu repositorio desde tu computadora usando la línea de comandos.

1.  Crea una nueva carpeta llamada `entregas`.

    ```bash
    mkdir entregas
    ```

2.  Mueve **todo el contenido que tenías antes** (tus notebooks, el `README.md`, etc.) a la nueva carpeta `entregas`.

    ```bash
    # USA ESTE COMANDO UNA SOLA VEZ
    # Moverá todos los archivos y carpetas (excepto los ocultos como .git) a 'entregas/'
    git mv * entregas/
    ```

    *Nota: Si el comando anterior da un error porque no puede mover la carpeta `entregas` dentro de sí misma, no te preocupes, es normal. Verifica que el resto de tus archivos se haya movido.*

3.  Agrega el repositorio original de la materia como un **submódulo**. Esto creará una carpeta `materiales_del_curso` que podrás actualizar para recibir nuevo contenido de la cátedra.

    ```bash
    # Este comando conecta tu repo con el de la materia
    git submodule add -f https://github.com/MetodosBayesianos/IBC1.2025.2.git materiales_del_curso
    ```

4.  Finalmente, guarda y sube todos estos cambios estructurales a GitHub.

    ```bash
    git commit -m "Reestructura a repo privado y agrega submódulo de materiales"
    git push
    ```

### Estructura Final

¡Listo\! Al terminar, la estructura de tu repositorio en tu computadora y en GitHub se verá así:

```
/tu_repositorio
|
|-- /entregas/
|   |-- 0-previa/
|   `-- README.md
|
|-- /materiales_del_curso/  <-- (Submódulo con los contenidos de la cátedra)
|
`-- .gitmodules               <-- (Archivo de configuración del submódulo)
```
