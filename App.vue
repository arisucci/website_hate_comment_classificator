<template>
  <div>
    <!-- Barra di navigazione con Bootstrap -->
    <nav class="navbar navbar-expand-lg navbar-dark bg-success">
      <div class="container-fluid">
        <!-- Logo o nome dell'applicazione senza link -->
        <span class="navbar-brand">Travel world</span>

        <!-- Pulsante di toggle per dispositivi mobili -->
        <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav" aria-controls="navbarNav" aria-expanded="false" aria-label="Toggle navigation">
          <span class="navbar-toggler-icon"></span>
        </button>

        <!-- Elementi della barra di navigazione -->
        <div class="collapse navbar-collapse" id="navbarNav">
          <ul class="navbar-nav">
            <li class="nav-item">
              <router-link class="nav-link" to="/profilo">Profilo</router-link>
            </li>
            <li class="nav-item">
              <router-link class="nav-link" to="/">Cerca</router-link>
            </li>
            <li class="nav-item">
              <router-link class="nav-link" to="/contatti">Contatti</router-link>
            </li>
          </ul>
        </div>
      </div>
    </nav>

    <!-- Visualizzazione del router -->
    <RouterView />
  </div>
</template>

<script setup>
import { RouterView } from 'vue-router'
</script>

<!-- Assicurati di aver importato Bootstrap nel tuo progetto -->
<style>
/* Aggiungi eventuali stili personalizzati qui */
</style>

