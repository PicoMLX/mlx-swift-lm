# Native fork Xcode Cloud setup

The post-clone hook enables package resolution for the IntegrationTesting project
and leaves it enabled for later scheme discovery and build steps.

Xcode Cloud initially sets `IDEPackageOnlyUseVersionsFromResolvedFile` and
`IDEDisableAutomaticPackageResolution` to true. Without a project-level resolved
file, even an explicit `xcodebuild -resolvePackageDependencies` then fails because
a resolved file is required. The hook resets both defaults to false.

The IntegrationTesting project declares remote packages in addition to the local
root package. Its graph is therefore a superset of the root `Package.resolved`.
The root lockfile cannot serve as its project lockfile. The fork resolves that
project graph in the post-clone hook instead of committing a generated file into
an upstream-owned project directory. Subsequent build steps reuse the generated
resolution; the defaults must stay enabled for those steps.

These settings and hooks belong to the PicoMLX fork. No upstream source or
project configuration is changed by this explanation.
