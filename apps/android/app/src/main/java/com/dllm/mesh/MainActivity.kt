package com.dllm.mesh

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Chat
import androidx.compose.material.icons.filled.Group
import androidx.compose.material.icons.filled.PieChart
import androidx.compose.material.icons.filled.QrCodeScanner
import androidx.compose.material.icons.filled.Settings
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.currentBackStackEntryAsState
import androidx.navigation.compose.rememberNavController
import com.dllm.mesh.ui.ChatScreen
import com.dllm.mesh.ui.NetworksScreen
import com.dllm.mesh.ui.PairingScreen
import com.dllm.mesh.ui.SettingsScreen
import com.dllm.mesh.ui.UsageScreen

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            DllmMeshTheme {
                DllmMeshApp()
            }
        }
    }
}

/** LAN-signal theme: ink panels, single teal accent, amber reserved for warnings. */
@Composable
fun DllmMeshTheme(content: @Composable () -> Unit) {
    MaterialTheme(
        colorScheme = darkColorScheme(
            background = Color(0xFF101418),
            surface = Color(0xFF171D24),
            primary = Color(0xFF2DD4BF),
            secondary = Color(0xFF93A1B0),
            tertiary = Color(0xFFF5B544),
            onBackground = Color(0xFFE8EDF2),
            onSurface = Color(0xFFE8EDF2),
        ),
        content = content,
    )
}

@Composable
fun DllmMeshApp() {
    val navController = rememberNavController()
    val backStack by navController.currentBackStackEntryAsState()
    val route = backStack?.destination?.route ?: "chat"

    Scaffold(
        containerColor = Color(0xFF101418),
        bottomBar = {
            NavigationBar(containerColor = Color(0xFF171D24)) {
                NavigationBarItem(
                    selected = route == "chat",
                    onClick = {
                        if (route != "chat") {
                            navController.navigate("chat") { launchSingleTop = true }
                        }
                    },
                    icon = { Icon(Icons.Filled.Chat, contentDescription = null) },
                    label = { Text("Chat") },
                )
                NavigationBarItem(
                    selected = route == "networks",
                    onClick = {
                        if (route != "networks") {
                            navController.navigate("networks") { launchSingleTop = true }
                        }
                    },
                    icon = { Icon(Icons.Filled.Group, contentDescription = null) },
                    label = { Text("Networks") },
                )
                NavigationBarItem(
                    selected = route == "pairing",
                    onClick = {
                        if (route != "pairing") {
                            navController.navigate("pairing") { launchSingleTop = true }
                        }
                    },
                    icon = { Icon(Icons.Filled.QrCodeScanner, contentDescription = null) },
                    label = { Text("Pairing") },
                )
                NavigationBarItem(
                    selected = route == "usage",
                    onClick = {
                        if (route != "usage") {
                            navController.navigate("usage") { launchSingleTop = true }
                        }
                    },
                    icon = { Icon(Icons.Filled.PieChart, contentDescription = null) },
                    label = { Text("Usage") },
                )
                NavigationBarItem(
                    selected = route == "settings",
                    onClick = {
                        if (route != "settings") {
                            navController.navigate("settings") { launchSingleTop = true }
                        }
                    },
                    icon = { Icon(Icons.Filled.Settings, contentDescription = null) },
                    label = { Text("Settings") },
                )
            }
        },
    ) { padding ->
        NavHost(
            navController = navController,
            startDestination = "chat",
            modifier = Modifier.padding(padding),
        ) {
            composable("chat") { ChatScreen() }
            composable("networks") { NetworksScreen() }
            composable("pairing") { PairingScreen() }
            composable("usage") { UsageScreen() }
            composable("settings") { SettingsScreen() }
        }
    }
}
